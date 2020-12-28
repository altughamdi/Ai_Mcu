#include "network_data.h"

ai_handle ai_network_data_weights_get(void)
{

  AI_ALIGNED(4)
  static const ai_u8 s_network_weights[ 1060 ] = {
    0x5d, 0x67, 0x1c, 0xbe, 0xca, 0x38, 0xbd, 0x3e, 0x64, 0x1d,
    0x01, 0xbf, 0x7a, 0x36, 0x85, 0xbe, 0x2e, 0x7f, 0xed, 0x3c,
    0xc6, 0x76, 0xb2, 0xbe, 0xd9, 0x63, 0x38, 0x3e, 0x2e, 0x3f,
    0xa2, 0x3e, 0xca, 0xa8, 0xd5, 0xbe, 0x90, 0x8e, 0x7a, 0xbe,
    0x00, 0xb0, 0x57, 0xbc, 0x68, 0xcf, 0xba, 0x3e, 0x2c, 0x1e,
    0x4b, 0xbe, 0xf8, 0x9e, 0x8f, 0xbe, 0xb8, 0xe8, 0xfa, 0x3e,
    0xd1, 0xc0, 0xbb, 0x3e, 0xd0, 0xc6, 0x14, 0xbe, 0x38, 0xba,
    0x20, 0xbe, 0x42, 0x5b, 0xd5, 0xbe, 0x7c, 0xe7, 0xc3, 0x3e,
    0x09, 0xae, 0xb7, 0x3e, 0x13, 0xf8, 0x98, 0xbd, 0x88, 0xca,
    0x08, 0xbf, 0x5d, 0x75, 0x73, 0x3e, 0x03, 0x84, 0x8d, 0x3e,
    0xe2, 0x30, 0xb4, 0x3d, 0x7d, 0x40, 0x9f, 0x3c, 0x9c, 0x6d,
    0x39, 0xbe, 0x8b, 0x5a, 0xa3, 0x3e, 0xd0, 0xcd, 0x19, 0xbf,
    0x5d, 0x05, 0x4a, 0x3e, 0x7e, 0xa6, 0xf6, 0x3e, 0x60, 0xb9,
    0xca, 0x3e, 0xc1, 0xcb, 0xc2, 0xbe, 0xa0, 0xbd, 0x9f, 0x3e,
    0xc5, 0xf4, 0x98, 0x3e, 0x5a, 0x33, 0x05, 0xbf, 0x58, 0xd0,
    0xf2, 0xbd, 0x44, 0xc3, 0x84, 0xbe, 0x46, 0x32, 0x09, 0xbf,
    0x98, 0x03, 0x17, 0xbe, 0x6f, 0xf3, 0x8e, 0xbe, 0xc2, 0x4d,
    0x2e, 0xbe, 0xd0, 0x19, 0x02, 0x3f, 0x0d, 0x8d, 0x81, 0x3e,
    0xbe, 0xa2, 0x7d, 0x3e, 0x87, 0x16, 0xcf, 0xbe, 0xab, 0xcf,
    0x35, 0xbe, 0xea, 0x1d, 0xec, 0xbd, 0x40, 0x4d, 0x95, 0x3e,
    0xdb, 0xf7, 0xf6, 0xbe, 0xd9, 0x95, 0x3b, 0x3e, 0xc8, 0x4b,
    0x7d, 0x3e, 0xc6, 0xd8, 0x24, 0xbf, 0x08, 0x59, 0xb5, 0x3e,
    0x84, 0xf8, 0x9a, 0x3e, 0x4c, 0xab, 0x0f, 0xbe, 0x00, 0xd0,
    0x17, 0xbe, 0xe4, 0x38, 0x39, 0xbe, 0x02, 0xdd, 0xc1, 0x3e,
    0xda, 0x51, 0x05, 0x3e, 0x5e, 0x79, 0x72, 0x3e, 0x71, 0x1d,
    0xa5, 0xbe, 0xa3, 0xfe, 0x17, 0x3e, 0x3e, 0x4c, 0x0c, 0xbd,
    0x07, 0xfd, 0x2f, 0xbd, 0x00, 0x00, 0x00, 0x00, 0x06, 0x3b,
    0x0e, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xa5, 0x73, 0xe7, 0x3d,
    0xd9, 0xef, 0x61, 0x3d, 0x65, 0x04, 0xec, 0xbd, 0x30, 0xa1,
    0x29, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xd9, 0x4a, 0x3e, 0x3e, 0x75, 0x1c, 0x56, 0xbc, 0xd3, 0x7b,
    0x4f, 0xbe, 0x00, 0x00, 0x00, 0x00, 0xc4, 0xdb, 0xd2, 0xbd,
    0xb7, 0x15, 0x9a, 0x3e, 0xbd, 0x2a, 0xa0, 0x3e, 0x58, 0x48,
    0x8e, 0xbe, 0x7d, 0x29, 0x92, 0x3e, 0x00, 0x8f, 0xec, 0xbe,
    0x39, 0xe0, 0xaf, 0xbe, 0x31, 0xfa, 0x07, 0x3e, 0x8d, 0xae,
    0xd0, 0x3d, 0x53, 0x54, 0xb9, 0xbd, 0x00, 0x50, 0x88, 0xbb,
    0x10, 0x68, 0x57, 0xbe, 0xfc, 0xed, 0x16, 0xbf, 0xc0, 0x7b,
    0xff, 0xbd, 0x73, 0xa2, 0xad, 0x3e, 0xc0, 0x01, 0xdd, 0x3d,
    0x5f, 0x76, 0x0a, 0xbf, 0x94, 0x7e, 0x08, 0xbe, 0x22, 0x64,
    0x9e, 0x3e, 0xc8, 0x55, 0x3c, 0x3e, 0x7d, 0x6f, 0xed, 0xbd,
    0xe0, 0x54, 0x74, 0x3d, 0xcb, 0xfb, 0x4b, 0x3d, 0xf7, 0x61,
    0xae, 0xbe, 0xf4, 0x25, 0x43, 0x3e, 0xb4, 0xbf, 0x87, 0xbe,
    0x84, 0xe8, 0xcd, 0xbe, 0x08, 0x6e, 0x05, 0xbe, 0xe1, 0xf1,
    0x03, 0x3f, 0xdc, 0xd7, 0xad, 0xbb, 0x36, 0x17, 0x94, 0x3e,
    0x6c, 0x9c, 0xf9, 0x3e, 0x71, 0xf6, 0x2a, 0xbe, 0xc1, 0x0d,
    0xc3, 0x3c, 0xf8, 0x62, 0x3f, 0x3e, 0x94, 0x21, 0xed, 0x3e,
    0xf8, 0x43, 0xb4, 0xbe, 0xa8, 0x28, 0x67, 0xbe, 0xa9, 0xcc,
    0xa1, 0xbe, 0x73, 0x59, 0xb2, 0x3d, 0x84, 0xb1, 0x09, 0xbe,
    0x4d, 0x87, 0x2f, 0x3e, 0x10, 0xe0, 0xd6, 0x3d, 0xc8, 0x3b,
    0x2a, 0x3e, 0x3e, 0x48, 0x80, 0xbe, 0xb7, 0x97, 0x43, 0x3e,
    0x12, 0xe0, 0xc9, 0x3e, 0x8c, 0xe8, 0x86, 0xbe, 0x90, 0x58,
    0x28, 0x3e, 0x66, 0xf9, 0x57, 0xbe, 0xbf, 0xbf, 0x81, 0xbd,
    0x38, 0x6f, 0x12, 0x3e, 0x4c, 0x53, 0x70, 0xbe, 0x08, 0x25,
    0xdb, 0xbe, 0xc7, 0xc6, 0x22, 0xbd, 0x39, 0xdb, 0xca, 0xbe,
    0xeb, 0xc9, 0x95, 0x3e, 0x62, 0xd9, 0x0b, 0x3f, 0x50, 0x24,
    0xb2, 0xbe, 0x00, 0xc0, 0xea, 0xbb, 0xe7, 0x5f, 0x8e, 0xbe,
    0x5d, 0xf8, 0x82, 0x3d, 0x5c, 0x52, 0xa8, 0xbc, 0x1c, 0x64,
    0x98, 0xbe, 0x36, 0x79, 0x2c, 0xbe, 0x41, 0x00, 0xda, 0x3e,
    0xb4, 0xa5, 0xea, 0xbe, 0x7c, 0x71, 0x98, 0xbe, 0x4c, 0x7f,
    0x29, 0xbf, 0x6c, 0xfa, 0x9d, 0xbe, 0xc1, 0x1e, 0x39, 0x3f,
    0xcf, 0xe4, 0xb2, 0x3d, 0x6b, 0x1a, 0xda, 0xbd, 0xcd, 0x97,
    0x0f, 0xbc, 0x00, 0x1c, 0xc1, 0x3c, 0xa0, 0xe5, 0x32, 0x3d,
    0x53, 0x35, 0x16, 0x3f, 0xd8, 0xcb, 0xa6, 0x3e, 0x3c, 0x75,
    0xf5, 0xbd, 0x58, 0x53, 0x0d, 0x3e, 0x35, 0xc9, 0x83, 0xbd,
    0xaa, 0xf7, 0xdf, 0xbe, 0xff, 0x31, 0xa1, 0x3e, 0xd4, 0xc3,
    0xb1, 0x3e, 0xe2, 0xe0, 0x6a, 0xbe, 0x54, 0x2b, 0xf2, 0x3e,
    0x3e, 0xd2, 0x90, 0x3e, 0xd0, 0x69, 0x2d, 0x3e, 0xfe, 0x93,
    0x9f, 0xbe, 0x64, 0x35, 0x45, 0x3d, 0xa0, 0x3b, 0x17, 0xbd,
    0xd0, 0x2a, 0xdc, 0x3d, 0xcf, 0x84, 0xc5, 0xbe, 0xa9, 0x87,
    0x91, 0x3e, 0x8e, 0x48, 0x9a, 0xbd, 0x78, 0x74, 0xc1, 0x3e,
    0xf0, 0x81, 0xdd, 0xbd, 0x6e, 0x97, 0xdc, 0x3e, 0xe9, 0x5b,
    0x25, 0x3f, 0xf0, 0xd9, 0xff, 0xbe, 0xab, 0x12, 0x99, 0x3e,
    0xf0, 0x04, 0x63, 0x3e, 0x6e, 0xd7, 0x01, 0xbf, 0x75, 0x3a,
    0x8a, 0x3c, 0xfa, 0xaf, 0x26, 0x3d, 0xa5, 0xfa, 0xea, 0x3d,
    0x00, 0x88, 0x62, 0x3d, 0x10, 0x23, 0x25, 0x3e, 0x44, 0xed,
    0xec, 0xbe, 0x9f, 0xdf, 0x70, 0x3e, 0xda, 0x89, 0x57, 0x3e,
    0x20, 0x0c, 0x02, 0xbe, 0xc9, 0xc5, 0x83, 0x3e, 0x7f, 0xf1,
    0xc0, 0x3e, 0x8f, 0xf5, 0xfc, 0xbc, 0xa4, 0x07, 0xf8, 0x3e,
    0xaa, 0xad, 0xb6, 0xbe, 0xc8, 0x74, 0x10, 0xbe, 0xd6, 0xc5,
    0x11, 0x3f, 0x22, 0x37, 0xff, 0xbe, 0xd3, 0xc9, 0x50, 0xbe,
    0x59, 0x98, 0x30, 0x3e, 0x00, 0xa5, 0xa8, 0x3d, 0x48, 0x1b,
    0x36, 0x3e, 0x8c, 0x57, 0x02, 0x3e, 0x62, 0xd6, 0x37, 0xbd,
    0x1b, 0x6b, 0x13, 0x3e, 0x68, 0x17, 0xa5, 0x3e, 0xde, 0x7a,
    0x5d, 0x3e, 0x20, 0x94, 0x5e, 0xbc, 0x3d, 0xfd, 0x35, 0xbe,
    0x29, 0xcc, 0xd1, 0x3c, 0x20, 0x68, 0xa9, 0x3c, 0x3f, 0x60,
    0x58, 0x3e, 0xcf, 0x7d, 0x11, 0xbe, 0xc5, 0x42, 0x99, 0xbd,
    0x43, 0xa1, 0x10, 0xbe, 0x5c, 0xd3, 0xd1, 0x3e, 0xad, 0x4e,
    0x1f, 0x3e, 0xb2, 0xd4, 0x88, 0x39, 0x3e, 0x0d, 0x72, 0x3e,
    0xb5, 0xdf, 0x37, 0xbf, 0xc7, 0x7e, 0xa0, 0xbc, 0x8d, 0x2a,
    0x30, 0x3f, 0x8c, 0x04, 0xd4, 0xbd, 0x18, 0xf0, 0x11, 0xbd,
    0x56, 0x0f, 0xc7, 0x3c, 0xc3, 0x9b, 0xae, 0x3e, 0x18, 0x7e,
    0x4e, 0x3e, 0xf2, 0x6a, 0x75, 0x3f, 0x6a, 0xf6, 0x55, 0xbe,
    0x76, 0xdb, 0xe4, 0xbe, 0x48, 0x82, 0x8d, 0xbd, 0x5a, 0xf1,
    0xe8, 0x3d, 0xd0, 0x4e, 0xff, 0x3e, 0x89, 0xd2, 0xe5, 0xbe,
    0xa6, 0x4d, 0xd0, 0xbd, 0x26, 0x89, 0xe9, 0xbe, 0x65, 0x03,
    0x19, 0xbe, 0xd8, 0xe0, 0x09, 0x3d, 0xf7, 0x7f, 0xd2, 0xbc,
    0x88, 0x73, 0x3f, 0x3e, 0xfb, 0x5a, 0x97, 0x3c, 0x47, 0xa5,
    0x82, 0xbe, 0x72, 0x07, 0x13, 0xbf, 0x25, 0x72, 0x26, 0x3f,
    0xe9, 0xaf, 0x0a, 0x3f, 0x79, 0xe4, 0xf7, 0x3d, 0x1b, 0x8f,
    0x05, 0x3f, 0x97, 0xb4, 0x8c, 0x3e, 0x52, 0x82, 0xc6, 0xbd,
    0x08, 0xd8, 0x3d, 0xbe, 0x96, 0x2e, 0x14, 0xbe, 0x1f, 0x93,
    0x62, 0xbe, 0xe7, 0x7e, 0x01, 0x3f, 0x74, 0xa7, 0xb2, 0x3e,
    0x96, 0xaf, 0x43, 0x3f, 0xc2, 0x64, 0x10, 0x3f, 0xb7, 0x07,
    0x09, 0x3f, 0x0a, 0x85, 0x91, 0x3e, 0x50, 0xfe, 0x18, 0x3f,
    0x44, 0x99, 0x91, 0xbe, 0x97, 0xd6, 0xe7, 0x3e, 0xfb, 0xa4,
    0x68, 0xbf, 0x09, 0x55, 0x00, 0x3f, 0xa6, 0x82, 0xd4, 0x3e
  };

  return AI_HANDLE_PTR(s_network_weights);

}

