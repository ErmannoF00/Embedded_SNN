"""Version 1: deterministic UART replay frame, IEEE CRC32, little-endian.

This is a HOST utility, NOT a complete Cyton ADC parser or MCU driver.
Frames carry a whole 1-second 250x8 float32 EEG window (microvolts).
"""
import struct,zlib
import numpy as np

MAGIC=b'ESNN'
N_RAW=250*8
PAYLOAD_BYTES=N_RAW*4
FRAME_BYTES=4+4+PAYLOAD_BYTES+4


def encode_window(sequence:int,window:np.ndarray)->bytes:
    if not 0 <= sequence < 2**32:raise ValueError('sequence outside uint32')
    x=np.asarray(window,dtype=np.float32)
    if x.shape!=(250,8) or not np.isfinite(x).all():
        raise ValueError('Expected finite (250,8) sample-major window')
    body=struct.pack('<I',sequence)+np.ascontiguousarray(x,dtype='<f4').tobytes()
    crc=zlib.crc32(body)&0xffffffff
    return MAGIC+body+struct.pack('<I',crc)


def decode_frame(data:bytes):
    if len(data)!=FRAME_BYTES or data[:4]!=MAGIC:
        raise ValueError('invalid frame size/magic')
    expected=struct.unpack_from('<I',data,FRAME_BYTES-4)[0]
    actual=zlib.crc32(data[4:-4])&0xffffffff
    if expected!=actual:raise ValueError('CRC mismatch')
    seq=struct.unpack_from('<I',data,4)[0]
    x=np.frombuffer(data[8:-4],dtype='<f4').reshape((250,8)).copy()
    if not np.isfinite(x).all():raise ValueError('non-finite EEG in frame')
    return seq,x
