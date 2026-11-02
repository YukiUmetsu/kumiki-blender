"""Dependency-free, genuinely valid PNG bytes for exporter/revision tests."""

import struct
import zlib


def one_pixel_png(rgb: tuple[int,int,int] = (255, 0, 0)) -> bytes:
    def chunk(tag: bytes, content: bytes) -> bytes:
        return (struct.pack(">I",len(content))+tag+content+
                struct.pack(">I",zlib.crc32(tag+content)))
    return (b"\x89PNG\r\n\x1a\n"+
            chunk(b"IHDR",struct.pack(">IIBBBBB",1,1,8,2,0,0,0))+
            chunk(b"IDAT",zlib.compress(b"\x00"+bytes(rgb)))+
            chunk(b"IEND",b""))
