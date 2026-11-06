#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace research_snn {
constexpr std::size_t N_IN=32, N_HID=48, N_OUT=2, N_SAMPLES=250, N_CHANNELS=8;
using Features=std::array<float,N_IN>;
using Window=std::array<float,N_SAMPLES*N_CHANNELS>;
using Scores=std::array<float,N_OUT>;
struct Weights {
  const float* w1;  // 48*32, row-major
  const float* b1;  // 48
  const float* w2;  // 2*48, row-major
  const float* b2;  // 2
};
struct Config {unsigned steps=20;float leak=.85f,adapt_decay=.90f,adapt_gain=.20f,threshold=1.f,readout_leak=.90f;};
enum class Mode {Full,Certified,Margin,Fixed};
struct Result {Scores scores{};unsigned steps=0,spikes=0;bool certified=false,valid=true;int pred=0;double lo=0,hi=0;};

inline bool extract_features(const Window& w,Features& f) {
  for(float v:w) if(!std::isfinite(v)) return false;
  for(std::size_t ch=0;ch<N_CHANNELS;++ch) {
    double square=0.,abs_sum=0.,diff_square=0.;unsigned crossings=0;
    for(std::size_t t=0;t<N_SAMPLES;++t) {
      double v=w[t*N_CHANNELS+ch];square+=v*v;abs_sum+=std::abs(v);
      if(t) {
        double prev=w[(t-1)*N_CHANNELS+ch];double delta=v-prev;
        diff_square+=delta*delta;
        crossings+=static_cast<unsigned>((v>=0)!=(prev>=0));
      }
    }
    f[ch*4+0]=static_cast<float>(std::sqrt(square/N_SAMPLES));
    f[ch*4+1]=static_cast<float>(abs_sum/N_SAMPLES);
    f[ch*4+2]=static_cast<float>(std::sqrt(diff_square/(N_SAMPLES-1)));
    f[ch*4+3]=static_cast<float>(static_cast<double>(crossings)/(N_SAMPLES-1));
  }
  return true;
}
inline bool normalize(Features& f,const float* mean,const float* stddev) {
  for(std::size_t i=0;i<N_IN;++i) {
    if(!std::isfinite(f[i]) || !std::isfinite(mean[i]) || !std::isfinite(stddev[i])) return false;
    const float den=std::max(stddev[i],1e-6f);
    f[i]=(f[i]-mean[i])/den;
    if(!std::isfinite(f[i])) return false;
  }
  return true;
}
inline Result infer_features(const Features& x,const Weights& p,Mode mode=Mode::Full,
                            Config cfg={},float threshold=1.0f,unsigned min_steps=2,
                            unsigned fixed_steps=20,double guard=1e-4) {
  Result result;
  if(cfg.steps==0 || cfg.readout_leak<0.f || cfg.readout_leak>1.f ||
     min_steps==0 || guard<0 || threshold<0 ||
     (mode==Mode::Fixed && (fixed_steps==0 || fixed_steps>cfg.steps))) {
    result.valid=false;return result;
  }
  for(float v:x) if(!std::isfinite(v)) {result.valid=false;return result;}
  std::array<float,N_HID> current{},mem{},adapt{};
  for(std::size_t j=0;j<N_HID;++j) {
    float v=p.b1[j];
    for(std::size_t i=0;i<N_IN;++i) v+=p.w1[j*N_IN+i]*x[i];
    if(!std::isfinite(v)) {result.valid=false;return result;}
    current[j]=v;
  }
  double lower_step=static_cast<double>(p.b2[1])-p.b2[0];
  double upper_step=lower_step;
  for(std::size_t j=0;j<N_HID;++j) {
    const double w=static_cast<double>(p.w2[N_HID+j])-p.w2[j];
    lower_step+=std::min(0.,w);upper_step+=std::max(0.,w);
  }
  const unsigned limit=mode==Mode::Fixed?fixed_steps:cfg.steps;
  for(unsigned t=1;t<=limit;++t) {
    std::array<float,N_HID> spikes{};
    for(std::size_t j=0;j<N_HID;++j) {
      const float candidate=cfg.leak*mem[j]+current[j]-cfg.adapt_gain*adapt[j];
      if(!std::isfinite(candidate)) {result.valid=false;return result;}
      const float s=candidate>=cfg.threshold?1.f:0.f;
      mem[j]=candidate-s;adapt[j]=cfg.adapt_decay*adapt[j]+s;
      if(!std::isfinite(adapt[j])) {result.valid=false;return result;}
      spikes[j]=s;result.spikes+=(s>0.f);
    }
    for(std::size_t k=0;k<N_OUT;++k) {
      float tmp=p.b2[k];
      for(std::size_t j=0;j<N_HID;++j) tmp+=p.w2[k*N_HID+j]*spikes[j];
      result.scores[k]=cfg.readout_leak*result.scores[k]+tmp;
      if(!std::isfinite(result.scores[k])) {result.valid=false;return result;}
    }
    result.steps=t;
    if(t<cfg.steps && t>=min_steps && mode==Mode::Certified) {
      const unsigned h=cfg.steps-t;
      const double decay=std::pow(static_cast<double>(cfg.readout_leak),h);
      const double geom=cfg.readout_leak==1.f?double(h):
                               (1.0-decay)/(1.0-static_cast<double>(cfg.readout_leak));
      const double margin=static_cast<double>(result.scores[1])-result.scores[0];
      const double center=decay*margin;
      result.lo=center+geom*lower_step;
      result.hi=center+geom*upper_step;
      if(result.lo>guard || result.hi<-guard) {
        result.certified=true;
        result.pred=result.lo>guard?1:0;
        return result;
      }
    }
    if(t<cfg.steps && t>=min_steps && mode==Mode::Margin &&
       std::abs(result.scores[1]-result.scores[0])>=threshold) {
      result.pred=result.scores[1]>result.scores[0]?1:0;
      return result;
    }
  }
  result.pred=result.scores[1]>result.scores[0]?1:0;
  result.certified=(mode==Mode::Certified && result.steps==cfg.steps);
  return result;
}
} // namespace research_snn
