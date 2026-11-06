#include "wire_protocol.hpp"
#include <cstdio>
int main() {
  research_snn::FrameParser parser;
  int c;
  while((c=std::getchar())!=EOF) {
    if(parser.consume(static_cast<std::uint8_t>(c))) {
      const auto& x=parser.window();
      std::printf("%u,%.8g,%.8g\n",parser.sequence(),x[0],x.back());
    }
  }
  std::fprintf(stderr,"good=%llu crc_errors=%llu invalid=%llu\n",
      static_cast<unsigned long long>(parser.good_frames()),
      static_cast<unsigned long long>(parser.crc_errors()),
      static_cast<unsigned long long>(parser.invalid_frames()));
  return 0;
}
