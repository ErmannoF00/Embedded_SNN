#include "research_runtime.hpp"
#include "weights.hpp"
#include <cstdio>

int main(int argc,char** argv) {
  using namespace research_snn;
  const Mode mode = argc>1 && argv[1][0]=='c'?Mode::Certified:Mode::Full;
  Weights p{model_weights::w1.data(),model_weights::b1.data(),
            model_weights::w2.data(),model_weights::b2.data()};
  Features x{};
  while(true) {
    if(std::scanf("%f",&x[0])!=1) break;
    for(unsigned i=1;i<N_IN;++i) if(std::scanf("%f",&x[i])!=1) return 2;
    const auto r=infer_features(x,p,mode);
    if(!r.valid) return 3;
    std::printf("%.9g,%.9g,%d,%u,%u\n",r.scores[0],r.scores[1],r.pred,r.steps,r.spikes);
  }
  return 0;
}
