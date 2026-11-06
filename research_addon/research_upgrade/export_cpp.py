"""Explicit NPZ-to-C++ export. Requires same weights as the trained SNN.

Only use checkpoints whose tensor orientation matches 48x32 and 2x48.
Do not check generated synthetic/random weights into public Git.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .certified_exit import Parameters


def export(weights_npz,header_path):
    p=Parameters.from_npz(str(weights_npz))
    if p.w1.shape!=(48,32):
        raise ValueError('C++ reference fixed to 48 hidden neurons')
    def array(name, arr):
        vals=','.join(format(float(v),'.9g')+'f' for v in arr.flatten())
        # `0f`, `-1f` are invalid float literals: enforce decimal point for integers
        def fmt(v):
            s=format(float(v),'.9g')
            if 'e' not in s and '.' not in s:s+='.'
            return s+'f'
        vals=',\n'.join('  '+', '.join(fmt(t) for t in arr.flatten()[k:k+8]) for k in range(0,arr.size,8))
        return f'inline constexpr std::array<float,{arr.size}> {name} = {{{{\n{vals}\n}}}};'
    sha=hashlib.sha256()
    for a in (p.w1,p.b1,p.w2,p.b2):sha.update(a.tobytes(order='C'))
    content='// MODEL WEIGHTS. SHA256 over ordered float32 W1,b1,W2,b2: '+sha.hexdigest()+'\n'
    content+='#pragma once\n#include <array>\nnamespace model_weights {\n'
    content+='\n'.join(array(k,a) for k,a in (('w1',p.w1),('b1',p.b1),('w2',p.w2),('b2',p.b2)))
    content+='\n} // namespace model_weights\n'
    output=Path(header_path);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(content)
    contract_path=Path(weights_npz).parent/'feature_contract.json'
    feature_contract=json.loads(contract_path.read_text()) if contract_path.exists() else None
    manifest={'feature_contract':feature_contract,
              'cxx_host_cli_input':'32 PRE-NORMALIZED features (not raw 250x8 windows)',
              'cxx_raw_window_extractor_compatible': (feature_contract is not None and
                  feature_contract.get('cxx_raw_window_extractor_supported') is True),
              'source_weights':str(weights_npz),'sha256_arrays':sha.hexdigest(),'output_header':str(output),
              'shapes':{'w1':[48,32],'b1':[48],'w2':[2,48],'b2':[2]}}
    Path(str(output)+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',required=True)
    parser.add_argument('--output',default='cpp/generated/weights.hpp')
    a=parser.parse_args();export(a.weights,a.output)
