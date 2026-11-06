import shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
from research_upgrade.wire_protocol import encode_window,decode_frame,FRAME_BYTES


def test_wire_round_trip():
    rng=np.random.default_rng(11)
    x=rng.normal(0,15,(250,8)).astype('f4')
    encoded=encode_window(4294967295,x)
    assert len(encoded)==FRAME_BYTES
    seq,y=decode_frame(encoded)
    assert seq==4294967295
    np.testing.assert_array_equal(x,y)
    bad=bytearray(encoded);bad[500]^=0x3
    with pytest.raises(ValueError,match='CRC'):
        decode_frame(bytes(bad))


def test_cpp_binary_parser_on_corrupt_stream(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('g++ missing')
    src=Path(__file__).resolve().parents[1]/'cpp'/'protocol_cli.cpp'
    exe=tmp_path/'parser'
    subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra',str(src),'-o',str(exe)],check=True)
    a=np.ones((250,8),'f4');b=np.full((250,8),-3.,'f4')
    valid_a=encode_window(7,a);valid_b=encode_window(9,b)
    corrupt=bytearray(encode_window(8,a));corrupt[600]^=1
    stream=b'junk'+valid_a+bytes(corrupt)+valid_b
    cp=subprocess.run([str(exe)],input=stream,capture_output=True,check=True)
    lines=cp.stdout.decode().strip().splitlines()
    assert lines==['7,1,1','9,-3,-3']
    assert 'good=2 crc_errors=1 invalid=0' in cp.stderr.decode()
