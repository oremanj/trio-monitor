# Vendored from tricycle

import attr
import codecs
import trio
from typing import Optional, Union, Tuple, AsyncIterator, TypeVar
from io import IncrementalNewlineDecoder


__all__ = ["BufferedReceiveStream", "TextReceiveStream"]

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class TextReceiveStream(trio.abc.AsyncResource):
    """Wraps a :class:`~trio.abc.ReceiveStream` with buffering and decoding
    capabilities for receiving line-oriented text.

    See :class:`io.TextIOWrapper` for more documentation on the ``encoding``,
    ``errors``, and ``newline`` arguments.

    Args:
      transport_stream (~trio.abc.ReceiveStream): The stream to receive
          data on.
      encoding (str): The encoding with which to decode received data.
      errors (str): Controls how to respond to decoding errors; common
          values include ``'strict'`` (throw an exception), ``'ignore'``
          (drop the bad character), or ``'replace'`` (replace the bad
          character with a replacement marker). The default of ``None``
          is equivalent to ``strict``.
      newline (str): Controls how line endings are handled. Use
          ``None`` to convert any newline format to ``'\n'``,
          ``''`` to accept any newline format and pass it through unchanged,
          or ``'\r'``, ``'\n'``, or ``'\r\n'`` to only accept that
          sequence as a newline.
      chunk_size (int): The number of bytes to request in each call to the
          underlying transport stream's
          :meth:`~trio.abc.ReceiveStream.receive_some` method.

    """

    transport_stream: trio.abc.ReceiveStream
    encoding: str
    chunk_size: int

    # Either _decoder is the same as _underlying_decoder, or _decoder
    # is an IncrementalNewlineDecoder. We need to remember both
    # because IncrementalNewlineDecoder doesn't have a .errors
    # attribute.
    _decoder: codecs.IncrementalDecoder
    _underlying_decoder: codecs.IncrementalDecoder

    def __init__(
        self,
        transport_stream: trio.abc.ReceiveStream,
        encoding: Optional[str] = None,
        *,
        errors: Optional[str] = None,
        newline: Optional[str] = "",
        chunk_size: int = 8192,
    ):
        if encoding is None:
            import locale

            encoding = locale.getpreferredencoding(False)

        self.transport_stream = transport_stream
        self.chunk_size = chunk_size
        self._encoding = encoding

        # The newline parameter is a newline sequence, or "" to accept
        # any of \r \n \r\n, or None to convert all to \n. self._newline
        # is the sequence we'll look for, or "" for any of \r \n \r\n.

        self._newline = newline if newline is not None else "\n"

        def make_decoder(
            encoding: str, errors: Optional[str], universal: bool, translate: bool
        ) -> codecs.IncrementalDecoder:
            info = codecs.lookup(encoding)
            decoder = info.incrementaldecoder(errors)  # type: ignore
            self._underlying_decoder = decoder
            if universal:
                return IncrementalNewlineDecoder(decoder, translate)  # type: ignore
            return decoder

        self._decoder = make_decoder(
            encoding, errors, newline is None or newline == "", newline is None
        )

        self._raw_chunk: Optional[bytes] = None

        # self._chunk[self._chunk_pos:] is the data that has been
        # passed through self._decoder but not yet returned from
        # receive_line().
        self._chunk = ""
        self._chunk_pos = 0

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def errors(self) -> Optional[str]:
        return self._underlying_decoder.errors

    @errors.setter
    def errors(self, value: Optional[str]) -> None:
        self._underlying_decoder.errors = value  # type: ignore

    @property
    def newlines(self) -> Union[str, Tuple[str, ...], None]:
        try:
            return self._decoder.newlines  # type: ignore
        except AttributeError:
            return None

    async def aclose(self) -> None:
        self._raw_chunk = None
        self._chunk = ""
        self._chunk_pos = 0
        await self.transport_stream.aclose()

    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over the lines in this stream."""
        while True:
            line = await self.receive_line()
            if line == "":
                return
            yield line

    async def receive_line(self, max_chars: int = -1) -> str:
        """Receive and decode data on this stream until ``max_chars`` have
        been received or a newline or end-of-file is encountered. The
        meaning of "newline" depends on the ``newline`` argument
        passed at construction time.

        Args:
          max_chars (int): The maximum number of characters to return if
              no newline sequence is received. If negative, read until
              newline or EOF.

        Returns:
          str: The line received. It always ends with a newline unless
              we reached ``max_chars`` or EOF. If there is no data left to
              return before EOF, returns an empty string (``""``).

        Raises:
          * UnicodeDecodeError: if the received data can't be decoded
          * Anything raised by the underlying transport stream's
            :meth:`~trio.abc.ReceiveStream.receive_some` method.

        """

        await trio.hazmat.checkpoint_if_cancelled()

        got_more = False
        line_end_pos = None
        while True:
            max_pos = len(self._chunk)
            if max_chars > 0:
                max_pos = min(max_pos, self._chunk_pos + max_chars)

            if self._newline == "":
                # Universal newlines without translation: search for any of \r,
                # \n, \r\n. Use of IncrementalNewlineDecoder ensures we never
                # split a \r\n sequence across two decoder outputs.
                crpos = self._chunk.find("\r", self._chunk_pos, max_pos)
                lfpos = self._chunk.find("\n", self._chunk_pos, max_pos)
                if crpos != -1 or lfpos != -1:
                    # Found a newline
                    if crpos != -1 and (lfpos == -1 or crpos < lfpos):
                        # CR exists and comes before LF.  LF may or
                        # may not exist.  If the first LF is one
                        # position after the first CR, we have a CRLF
                        # and must end the line after the entire CRLF
                        # sequence. Otherwise, end after the CR.
                        line_end_pos = crpos + 1 + (lfpos == crpos + 1)
                    else:
                        # CR either does not exist or comes after LF,
                        # so this line is delimited by LF.
                        line_end_pos = lfpos + 1
                    break
            else:
                # Just need to end on occurrences of self._newline.
                # (If we're using universal newlines with translation, we
                # set it to "\n" in the constructor.)
                nlpos = self._chunk.find(self._newline, self._chunk_pos, max_pos)
                if nlpos != -1:
                    line_end_pos = nlpos + len(self._newline)
                    break

            # If we found a newline in self._chunk, we broke out of the
            # loop above. Getting here means we either need more data or
            # hit our max_chars limit and must return without the newline.
            if max_pos == self._chunk_pos + max_chars:
                # Hit limit, return what we've got.
                line_end_pos = max_pos
                break

            # Need to pull down more raw data to decode
            if self._raw_chunk is None:
                self._raw_chunk = await self.transport_stream.receive_some(
                    self.chunk_size
                )
                got_more = True

            if self._raw_chunk == b"":
                # EOF on underlying stream. Pull out whatever the decoder
                # has left for us; if that's nothing, return EOF ourselves.
                chunk = self._decoder.decode(self._raw_chunk, final=True)
                if not chunk:
                    line_end_pos = len(self._chunk)
                    break
            else:
                chunk = self._decoder.decode(self._raw_chunk)

            # We need to reallocate self._chunk in order to append the new
            # stuff, so we'll throw away already-consumed output while we're
            # at it. We don't do this at every call to receive_line() because
            # it would result in quadratic-time performance with short lines.
            # (We still get quadratic-time performance with arbitrarily long
            # lines, but we'll not worry about that for now.)
            self._chunk = self._chunk[self._chunk_pos :] + chunk
            self._chunk_pos = 0

            # We've incorporated _raw_chunk into _chunk, so null it out.
            # If decoding failed we would leave _raw_chunk non-null and
            # try again to decode it on a future call, maybe with a different
            # errors parameter.
            self._raw_chunk = None

        # We break out of the loop when we find the point we want to
        # chop at.  All that's left is to return it to the caller.

        if not got_more:
            # If we never called receive_some(), we only did half a checkpoint,
            # and need to do the other half before returning.
            await trio.hazmat.cancel_shielded_checkpoint()

        ret = self._chunk[self._chunk_pos : line_end_pos]
        self._chunk_pos = line_end_pos

        # If we're consuming the whole buffer, compact it now since
        # that's basically free. Otherwise wait until we next pull
        # down a chunk, so we don't have too poor performance when
        # receiving lots of short lines.
        if self._chunk_pos == len(self._chunk):
            self._chunk = ""
            self._chunk_pos = 0

        return ret
