#pragma once
// No-allocation byte-stream decoder for ESNN v1 replay frames.
// Target UART ISR/DMA integration, buffering and timing are NOT supplied.
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "research_runtime.hpp"

namespace research_snn {
constexpr std::size_t WIRE_PAYLOAD=N_SAMPLES*N_CHANNELS*4;
constexpr std::size_t WIRE_FRAME_SIZE=4+4+WIRE_PAYLOAD+4;

class FrameParser {
public:
  FrameParser()=default;
  bool consume(std::uint8_t b) {
    // A successful decode leaves the completed window accessible until
    // more data is consumed. Do not call from ISR and thread concurrently.
    if(state_==State::Magic) {
      constexpr std::uint8_t magic[4]={'E','S','N','N'};
      if(b==magic[magic_pos_]) {++magic_pos_;if(magic_pos_==4) {
        state_=State::Seq;magic_pos_=0;pos_=0;crc_=0xffffffffu;
      }} else magic_pos_=(b==magic[0])?1:0;
      return false;
    }
    if(state_==State::Seq) {
      seq_bytes_[pos_++]=b;crc_update(b);
      if(pos_==4) {state_=State::Payload;pos_=0;}
      return false;
    }
    if(state_==State::Payload) {
      payload_[pos_++]=b;crc_update(b);
      if(pos_==WIRE_PAYLOAD) {state_=State::CRC;pos_=0;}
      return false;
    }
    crc_bytes_[pos_++]=b;
    if(pos_!=4)return false;
    state_=State::Magic;pos_=0;
    std::uint32_t rxcrc=load_le32(crc_bytes_.data());
    if(rxcrc!=(crc_^0xffffffffu)) {++crc_errors_;return false;}
    for(std::size_t i=0;i<N_SAMPLES*N_CHANNELS;++i) {
      const std::uint32_t u=load_le32(payload_.data()+4*i);
      float value;
      static_assert(sizeof(float)==sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
                    "requires IEEE-754 binary32 float");
      std::memcpy(&value,&u,4);
      if(!std::isfinite(value)) {++invalid_frames_;return false;}
      window_[i]=value;
    }
    sequence_=load_le32(seq_bytes_.data());
    ++good_frames_;return true;
  }
  const Window& window()const{return window_;}
  std::uint32_t sequence()const{return sequence_;}
  std::uint64_t good_frames()const{return good_frames_;}
  std::uint64_t crc_errors()const{return crc_errors_;}
  std::uint64_t invalid_frames()const{return invalid_frames_;}
private:
  enum class State {Magic,Seq,Payload,CRC};
  static std::uint32_t load_le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])|
           (static_cast<std::uint32_t>(p[1])<<8)|
           (static_cast<std::uint32_t>(p[2])<<16)|
           (static_cast<std::uint32_t>(p[3])<<24);
  }
  void crc_update(std::uint8_t data) {
    crc_^=data;
    for(int k=0;k<8;++k)crc_=(crc_>>1)^((crc_&1u)?0xedb88320u:0u);
  }
  State state_=State::Magic;
  std::size_t magic_pos_=0,pos_=0;
  std::uint32_t crc_=0xffffffffu,sequence_=0;
  std::array<std::uint8_t,4> seq_bytes_{},crc_bytes_{};
  std::array<std::uint8_t,WIRE_PAYLOAD> payload_{};
  Window window_{};
  std::uint64_t good_frames_=0,crc_errors_=0,invalid_frames_=0;
};
} // namespace research_snn
