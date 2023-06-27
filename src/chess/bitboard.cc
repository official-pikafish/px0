/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "chess/bitboard.h"

#include "utils/exception.h"

namespace lczero {

namespace {

const Move kIdxToMove[] = {
    "a0a1", "a0a2", "a0a3", "a0a4", "a0a5", "a0a6", "a0a7", "a0a8", "a0a9", "a0b0", "a0b2", "a0c0", "a0c1", "a0d0", "a0e0",
    "a0f0", "a0g0", "a0h0", "a0i0", "a1a0", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "a1a9", "a1b1", "a1b3",
    "a1c0", "a1c1", "a1c2", "a1d1", "a1e1", "a1f1", "a1g1", "a1h1", "a1i1", "a2a0", "a2a1", "a2a3", "a2a4", "a2a5", "a2a6",
    "a2a7", "a2a8", "a2a9", "a2b0", "a2b2", "a2b4", "a2c0", "a2c1", "a2c2", "a2c3", "a2c4", "a2d2", "a2e2", "a2f2", "a2g2",
    "a2h2", "a2i2", "a3a0", "a3a1", "a3a2", "a3a4", "a3a5", "a3a6", "a3a7", "a3a8", "a3a9", "a3b1", "a3b3", "a3b5", "a3c2",
    "a3c3", "a3c4", "a3d3", "a3e3", "a3f3", "a3g3", "a3h3", "a3i3", "a4a0", "a4a1", "a4a2", "a4a3", "a4a5", "a4a6", "a4a7",
    "a4a8", "a4a9", "a4b2", "a4b4", "a4b6", "a4c3", "a4c4", "a4c5", "a4d4", "a4e4", "a4f4", "a4g4", "a4h4", "a4i4", "a5a0",
    "a5a1", "a5a2", "a5a3", "a5a4", "a5a6", "a5a7", "a5a8", "a5a9", "a5b3", "a5b5", "a5b7", "a5c4", "a5c5", "a5c6", "a5d5",
    "a5e5", "a5f5", "a5g5", "a5h5", "a5i5", "a6a0", "a6a1", "a6a2", "a6a3", "a6a4", "a6a5", "a6a7", "a6a8", "a6a9", "a6b4",
    "a6b6", "a6b8", "a6c5", "a6c6", "a6c7", "a6d6", "a6e6", "a6f6", "a6g6", "a6h6", "a6i6", "a7a0", "a7a1", "a7a2", "a7a3",
    "a7a4", "a7a5", "a7a6", "a7a8", "a7a9", "a7b5", "a7b7", "a7b9", "a7c6", "a7c7", "a7c8", "a7d7", "a7e7", "a7f7", "a7g7",
    "a7h7", "a7i7", "a8a0", "a8a1", "a8a2", "a8a3", "a8a4", "a8a5", "a8a6", "a8a7", "a8a9", "a8b6", "a8b8", "a8c7", "a8c8",
    "a8c9", "a8d8", "a8e8", "a8f8", "a8g8", "a8h8", "a8i8", "a9a0", "a9a1", "a9a2", "a9a3", "a9a4", "a9a5", "a9a6", "a9a7",
    "a9a8", "a9b7", "a9b9", "a9c8", "a9c9", "a9d9", "a9e9", "a9f9", "a9g9", "a9h9", "a9i9", "b0a0", "b0a2", "b0b1", "b0b2",
    "b0b3", "b0b4", "b0b5", "b0b6", "b0b7", "b0b8", "b0b9", "b0c0", "b0c2", "b0d0", "b0d1", "b0e0", "b0f0", "b0g0", "b0h0",
    "b0i0", "b1a1", "b1a3", "b1b0", "b1b2", "b1b3", "b1b4", "b1b5", "b1b6", "b1b7", "b1b8", "b1b9", "b1c1", "b1c3", "b1d0",
    "b1d1", "b1d2", "b1e1", "b1f1", "b1g1", "b1h1", "b1i1", "b2a0", "b2a2", "b2a4", "b2b0", "b2b1", "b2b3", "b2b4", "b2b5",
    "b2b6", "b2b7", "b2b8", "b2b9", "b2c0", "b2c2", "b2c4", "b2d1", "b2d2", "b2d3", "b2e2", "b2f2", "b2g2", "b2h2", "b2i2",
    "b3a1", "b3a3", "b3a5", "b3b0", "b3b1", "b3b2", "b3b4", "b3b5", "b3b6", "b3b7", "b3b8", "b3b9", "b3c1", "b3c3", "b3c5",
    "b3d2", "b3d3", "b3d4", "b3e3", "b3f3", "b3g3", "b3h3", "b3i3", "b4a2", "b4a4", "b4a6", "b4b0", "b4b1", "b4b2", "b4b3",
    "b4b5", "b4b6", "b4b7", "b4b8", "b4b9", "b4c2", "b4c4", "b4c6", "b4d3", "b4d4", "b4d5", "b4e4", "b4f4", "b4g4", "b4h4",
    "b4i4", "b5a3", "b5a5", "b5a7", "b5b0", "b5b1", "b5b2", "b5b3", "b5b4", "b5b6", "b5b7", "b5b8", "b5b9", "b5c3", "b5c5",
    "b5c7", "b5d4", "b5d5", "b5d6", "b5e5", "b5f5", "b5g5", "b5h5", "b5i5", "b6a4", "b6a6", "b6a8", "b6b0", "b6b1", "b6b2",
    "b6b3", "b6b4", "b6b5", "b6b7", "b6b8", "b6b9", "b6c4", "b6c6", "b6c8", "b6d5", "b6d6", "b6d7", "b6e6", "b6f6", "b6g6",
    "b6h6", "b6i6", "b7a5", "b7a7", "b7a9", "b7b0", "b7b1", "b7b2", "b7b3", "b7b4", "b7b5", "b7b6", "b7b8", "b7b9", "b7c5",
    "b7c7", "b7c9", "b7d6", "b7d7", "b7d8", "b7e7", "b7f7", "b7g7", "b7h7", "b7i7", "b8a6", "b8a8", "b8b0", "b8b1", "b8b2",
    "b8b3", "b8b4", "b8b5", "b8b6", "b8b7", "b8b9", "b8c6", "b8c8", "b8d7", "b8d8", "b8d9", "b8e8", "b8f8", "b8g8", "b8h8",
    "b8i8", "b9a7", "b9a9", "b9b0", "b9b1", "b9b2", "b9b3", "b9b4", "b9b5", "b9b6", "b9b7", "b9b8", "b9c7", "b9c9", "b9d8",
    "b9d9", "b9e9", "b9f9", "b9g9", "b9h9", "b9i9", "c0a0", "c0a1", "c0a2", "c0b0", "c0b2", "c0c1", "c0c2", "c0c3", "c0c4",
    "c0c5", "c0c6", "c0c7", "c0c8", "c0c9", "c0d0", "c0d2", "c0e0", "c0e1", "c0e2", "c0f0", "c0g0", "c0h0", "c0i0", "c1a0",
    "c1a1", "c1a2", "c1b1", "c1b3", "c1c0", "c1c2", "c1c3", "c1c4", "c1c5", "c1c6", "c1c7", "c1c8", "c1c9", "c1d1", "c1d3",
    "c1e0", "c1e1", "c1e2", "c1f1", "c1g1", "c1h1", "c1i1", "c2a1", "c2a2", "c2a3", "c2b0", "c2b2", "c2b4", "c2c0", "c2c1",
    "c2c3", "c2c4", "c2c5", "c2c6", "c2c7", "c2c8", "c2c9", "c2d0", "c2d2", "c2d4", "c2e1", "c2e2", "c2e3", "c2f2", "c2g2",
    "c2h2", "c2i2", "c3a2", "c3a3", "c3a4", "c3b1", "c3b3", "c3b5", "c3c0", "c3c1", "c3c2", "c3c4", "c3c5", "c3c6", "c3c7",
    "c3c8", "c3c9", "c3d1", "c3d3", "c3d5", "c3e2", "c3e3", "c3e4", "c3f3", "c3g3", "c3h3", "c3i3", "c4a2", "c4a3", "c4a4",
    "c4a5", "c4b2", "c4b4", "c4b6", "c4c0", "c4c1", "c4c2", "c4c3", "c4c5", "c4c6", "c4c7", "c4c8", "c4c9", "c4d2", "c4d4",
    "c4d6", "c4e2", "c4e3", "c4e4", "c4e5", "c4f4", "c4g4", "c4h4", "c4i4", "c5a4", "c5a5", "c5a6", "c5b3", "c5b5", "c5b7",
    "c5c0", "c5c1", "c5c2", "c5c3", "c5c4", "c5c6", "c5c7", "c5c8", "c5c9", "c5d3", "c5d5", "c5d7", "c5e4", "c5e5", "c5e6",
    "c5f5", "c5g5", "c5h5", "c5i5", "c6a5", "c6a6", "c6a7", "c6b4", "c6b6", "c6b8", "c6c0", "c6c1", "c6c2", "c6c3", "c6c4",
    "c6c5", "c6c7", "c6c8", "c6c9", "c6d4", "c6d6", "c6d8", "c6e5", "c6e6", "c6e7", "c6f6", "c6g6", "c6h6", "c6i6", "c7a6",
    "c7a7", "c7a8", "c7b5", "c7b7", "c7b9", "c7c0", "c7c1", "c7c2", "c7c3", "c7c4", "c7c5", "c7c6", "c7c8", "c7c9", "c7d5",
    "c7d7", "c7d9", "c7e6", "c7e7", "c7e8", "c7f7", "c7g7", "c7h7", "c7i7", "c8a7", "c8a8", "c8a9", "c8b6", "c8b8", "c8c0",
    "c8c1", "c8c2", "c8c3", "c8c4", "c8c5", "c8c6", "c8c7", "c8c9", "c8d6", "c8d8", "c8e7", "c8e8", "c8e9", "c8f8", "c8g8",
    "c8h8", "c8i8", "c9a8", "c9a9", "c9b7", "c9b9", "c9c0", "c9c1", "c9c2", "c9c3", "c9c4", "c9c5", "c9c6", "c9c7", "c9c8",
    "c9d7", "c9d9", "c9e8", "c9e9", "c9f9", "c9g9", "c9h9", "c9i9", "d0a0", "d0b0", "d0b1", "d0c0", "d0c2", "d0d1", "d0d2",
    "d0d3", "d0d4", "d0d5", "d0d6", "d0d7", "d0d8", "d0d9", "d0e0", "d0e1", "d0e2", "d0f0", "d0f1", "d0g0", "d0h0", "d0i0",
    "d1a1", "d1b0", "d1b1", "d1b2", "d1c1", "d1c3", "d1d0", "d1d2", "d1d3", "d1d4", "d1d5", "d1d6", "d1d7", "d1d8", "d1d9",
    "d1e1", "d1e3", "d1f0", "d1f1", "d1f2", "d1g1", "d1h1", "d1i1", "d2a2", "d2b1", "d2b2", "d2b3", "d2c0", "d2c2", "d2c4",
    "d2d0", "d2d1", "d2d3", "d2d4", "d2d5", "d2d6", "d2d7", "d2d8", "d2d9", "d2e0", "d2e1", "d2e2", "d2e4", "d2f1", "d2f2",
    "d2f3", "d2g2", "d2h2", "d2i2", "d3a3", "d3b2", "d3b3", "d3b4", "d3c1", "d3c3", "d3c5", "d3d0", "d3d1", "d3d2", "d3d4",
    "d3d5", "d3d6", "d3d7", "d3d8", "d3d9", "d3e1", "d3e3", "d3e5", "d3f2", "d3f3", "d3f4", "d3g3", "d3h3", "d3i3", "d4a4",
    "d4b3", "d4b4", "d4b5", "d4c2", "d4c4", "d4c6", "d4d0", "d4d1", "d4d2", "d4d3", "d4d5", "d4d6", "d4d7", "d4d8", "d4d9",
    "d4e2", "d4e4", "d4e6", "d4f3", "d4f4", "d4f5", "d4g4", "d4h4", "d4i4", "d5a5", "d5b4", "d5b5", "d5b6", "d5c3", "d5c5",
    "d5c7", "d5d0", "d5d1", "d5d2", "d5d3", "d5d4", "d5d6", "d5d7", "d5d8", "d5d9", "d5e3", "d5e5", "d5e7", "d5f4", "d5f5",
    "d5f6", "d5g5", "d5h5", "d5i5", "d6a6", "d6b5", "d6b6", "d6b7", "d6c4", "d6c6", "d6c8", "d6d0", "d6d1", "d6d2", "d6d3",
    "d6d4", "d6d5", "d6d7", "d6d8", "d6d9", "d6e4", "d6e6", "d6e8", "d6f5", "d6f6", "d6f7", "d6g6", "d6h6", "d6i6", "d7a7",
    "d7b6", "d7b7", "d7b8", "d7c5", "d7c7", "d7c9", "d7d0", "d7d1", "d7d2", "d7d3", "d7d4", "d7d5", "d7d6", "d7d8", "d7d9",
    "d7e5", "d7e7", "d7e9", "d7f6", "d7f7", "d7f8", "d7g7", "d7h7", "d7i7", "d8a8", "d8b7", "d8b8", "d8b9", "d8c6", "d8c8",
    "d8d0", "d8d1", "d8d2", "d8d3", "d8d4", "d8d5", "d8d6", "d8d7", "d8d9", "d8e6", "d8e8", "d8f7", "d8f8", "d8f9", "d8g8",
    "d8h8", "d8i8", "d9a9", "d9b8", "d9b9", "d9c7", "d9c9", "d9d0", "d9d1", "d9d2", "d9d3", "d9d4", "d9d5", "d9d6", "d9d7",
    "d9d8", "d9e7", "d9e9", "d9f8", "d9f9", "d9g9", "d9h9", "d9i9", "e0a0", "e0b0", "e0c0", "e0c1", "e0d0", "e0d2", "e0e1",
    "e0e2", "e0e3", "e0e4", "e0e5", "e0e6", "e0e7", "e0e8", "e0e9", "e0f0", "e0f2", "e0g0", "e0g1", "e0h0", "e0i0", "e1a1",
    "e1b1", "e1c0", "e1c1", "e1c2", "e1d0", "e1d1", "e1d2", "e1d3", "e1e0", "e1e2", "e1e3", "e1e4", "e1e5", "e1e6", "e1e7",
    "e1e8", "e1e9", "e1f0", "e1f1", "e1f2", "e1f3", "e1g0", "e1g1", "e1g2", "e1h1", "e1i1", "e2a2", "e2b2", "e2c0", "e2c1",
    "e2c2", "e2c3", "e2c4", "e2d0", "e2d2", "e2d4", "e2e0", "e2e1", "e2e3", "e2e4", "e2e5", "e2e6", "e2e7", "e2e8", "e2e9",
    "e2f0", "e2f2", "e2f4", "e2g0", "e2g1", "e2g2", "e2g3", "e2g4", "e2h2", "e2i2", "e3a3", "e3b3", "e3c2", "e3c3", "e3c4",
    "e3d1", "e3d3", "e3d5", "e3e0", "e3e1", "e3e2", "e3e4", "e3e5", "e3e6", "e3e7", "e3e8", "e3e9", "e3f1", "e3f3", "e3f5",
    "e3g2", "e3g3", "e3g4", "e3h3", "e3i3", "e4a4", "e4b4", "e4c3", "e4c4", "e4c5", "e4d2", "e4d4", "e4d6", "e4e0", "e4e1",
    "e4e2", "e4e3", "e4e5", "e4e6", "e4e7", "e4e8", "e4e9", "e4f2", "e4f4", "e4f6", "e4g3", "e4g4", "e4g5", "e4h4", "e4i4",
    "e5a5", "e5b5", "e5c4", "e5c5", "e5c6", "e5d3", "e5d5", "e5d7", "e5e0", "e5e1", "e5e2", "e5e3", "e5e4", "e5e6", "e5e7",
    "e5e8", "e5e9", "e5f3", "e5f5", "e5f7", "e5g4", "e5g5", "e5g6", "e5h5", "e5i5", "e6a6", "e6b6", "e6c5", "e6c6", "e6c7",
    "e6d4", "e6d6", "e6d8", "e6e0", "e6e1", "e6e2", "e6e3", "e6e4", "e6e5", "e6e7", "e6e8", "e6e9", "e6f4", "e6f6", "e6f8",
    "e6g5", "e6g6", "e6g7", "e6h6", "e6i6", "e7a7", "e7b7", "e7c6", "e7c7", "e7c8", "e7d5", "e7d7", "e7d9", "e7e0", "e7e1",
    "e7e2", "e7e3", "e7e4", "e7e5", "e7e6", "e7e8", "e7e9", "e7f5", "e7f7", "e7f9", "e7g6", "e7g7", "e7g8", "e7h7", "e7i7",
    "e8a8", "e8b8", "e8c7", "e8c8", "e8c9", "e8d6", "e8d8", "e8e0", "e8e1", "e8e2", "e8e3", "e8e4", "e8e5", "e8e6", "e8e7",
    "e8e9", "e8f6", "e8f8", "e8g7", "e8g8", "e8g9", "e8h8", "e8i8", "e9a9", "e9b9", "e9c8", "e9c9", "e9d7", "e9d9", "e9e0",
    "e9e1", "e9e2", "e9e3", "e9e4", "e9e5", "e9e6", "e9e7", "e9e8", "e9f7", "e9f9", "e9g8", "e9g9", "e9h9", "e9i9", "f0a0",
    "f0b0", "f0c0", "f0d0", "f0d1", "f0e0", "f0e1", "f0e2", "f0f1", "f0f2", "f0f3", "f0f4", "f0f5", "f0f6", "f0f7", "f0f8",
    "f0f9", "f0g0", "f0g2", "f0h0", "f0h1", "f0i0", "f1a1", "f1b1", "f1c1", "f1d0", "f1d1", "f1d2", "f1e1", "f1e3", "f1f0",
    "f1f2", "f1f3", "f1f4", "f1f5", "f1f6", "f1f7", "f1f8", "f1f9", "f1g1", "f1g3", "f1h0", "f1h1", "f1h2", "f1i1", "f2a2",
    "f2b2", "f2c2", "f2d1", "f2d2", "f2d3", "f2e0", "f2e1", "f2e2", "f2e4", "f2f0", "f2f1", "f2f3", "f2f4", "f2f5", "f2f6",
    "f2f7", "f2f8", "f2f9", "f2g0", "f2g2", "f2g4", "f2h1", "f2h2", "f2h3", "f2i2", "f3a3", "f3b3", "f3c3", "f3d2", "f3d3",
    "f3d4", "f3e1", "f3e3", "f3e5", "f3f0", "f3f1", "f3f2", "f3f4", "f3f5", "f3f6", "f3f7", "f3f8", "f3f9", "f3g1", "f3g3",
    "f3g5", "f3h2", "f3h3", "f3h4", "f3i3", "f4a4", "f4b4", "f4c4", "f4d3", "f4d4", "f4d5", "f4e2", "f4e4", "f4e6", "f4f0",
    "f4f1", "f4f2", "f4f3", "f4f5", "f4f6", "f4f7", "f4f8", "f4f9", "f4g2", "f4g4", "f4g6", "f4h3", "f4h4", "f4h5", "f4i4",
    "f5a5", "f5b5", "f5c5", "f5d4", "f5d5", "f5d6", "f5e3", "f5e5", "f5e7", "f5f0", "f5f1", "f5f2", "f5f3", "f5f4", "f5f6",
    "f5f7", "f5f8", "f5f9", "f5g3", "f5g5", "f5g7", "f5h4", "f5h5", "f5h6", "f5i5", "f6a6", "f6b6", "f6c6", "f6d5", "f6d6",
    "f6d7", "f6e4", "f6e6", "f6e8", "f6f0", "f6f1", "f6f2", "f6f3", "f6f4", "f6f5", "f6f7", "f6f8", "f6f9", "f6g4", "f6g6",
    "f6g8", "f6h5", "f6h6", "f6h7", "f6i6", "f7a7", "f7b7", "f7c7", "f7d6", "f7d7", "f7d8", "f7e5", "f7e7", "f7e9", "f7f0",
    "f7f1", "f7f2", "f7f3", "f7f4", "f7f5", "f7f6", "f7f8", "f7f9", "f7g5", "f7g7", "f7g9", "f7h6", "f7h7", "f7h8", "f7i7",
    "f8a8", "f8b8", "f8c8", "f8d7", "f8d8", "f8d9", "f8e6", "f8e8", "f8f0", "f8f1", "f8f2", "f8f3", "f8f4", "f8f5", "f8f6",
    "f8f7", "f8f9", "f8g6", "f8g8", "f8h7", "f8h8", "f8h9", "f8i8", "f9a9", "f9b9", "f9c9", "f9d8", "f9d9", "f9e7", "f9e9",
    "f9f0", "f9f1", "f9f2", "f9f3", "f9f4", "f9f5", "f9f6", "f9f7", "f9f8", "f9g7", "f9g9", "f9h8", "f9h9", "f9i9", "g0a0",
    "g0b0", "g0c0", "g0d0", "g0e0", "g0e1", "g0e2", "g0f0", "g0f2", "g0g1", "g0g2", "g0g3", "g0g4", "g0g5", "g0g6", "g0g7",
    "g0g8", "g0g9", "g0h0", "g0h2", "g0i0", "g0i1", "g0i2", "g1a1", "g1b1", "g1c1", "g1d1", "g1e0", "g1e1", "g1e2", "g1f1",
    "g1f3", "g1g0", "g1g2", "g1g3", "g1g4", "g1g5", "g1g6", "g1g7", "g1g8", "g1g9", "g1h1", "g1h3", "g1i0", "g1i1", "g1i2",
    "g2a2", "g2b2", "g2c2", "g2d2", "g2e1", "g2e2", "g2e3", "g2f0", "g2f2", "g2f4", "g2g0", "g2g1", "g2g3", "g2g4", "g2g5",
    "g2g6", "g2g7", "g2g8", "g2g9", "g2h0", "g2h2", "g2h4", "g2i1", "g2i2", "g2i3", "g3a3", "g3b3", "g3c3", "g3d3", "g3e2",
    "g3e3", "g3e4", "g3f1", "g3f3", "g3f5", "g3g0", "g3g1", "g3g2", "g3g4", "g3g5", "g3g6", "g3g7", "g3g8", "g3g9", "g3h1",
    "g3h3", "g3h5", "g3i2", "g3i3", "g3i4", "g4a4", "g4b4", "g4c4", "g4d4", "g4e2", "g4e3", "g4e4", "g4e5", "g4f2", "g4f4",
    "g4f6", "g4g0", "g4g1", "g4g2", "g4g3", "g4g5", "g4g6", "g4g7", "g4g8", "g4g9", "g4h2", "g4h4", "g4h6", "g4i2", "g4i3",
    "g4i4", "g4i5", "g5a5", "g5b5", "g5c5", "g5d5", "g5e4", "g5e5", "g5e6", "g5f3", "g5f5", "g5f7", "g5g0", "g5g1", "g5g2",
    "g5g3", "g5g4", "g5g6", "g5g7", "g5g8", "g5g9", "g5h3", "g5h5", "g5h7", "g5i4", "g5i5", "g5i6", "g6a6", "g6b6", "g6c6",
    "g6d6", "g6e5", "g6e6", "g6e7", "g6f4", "g6f6", "g6f8", "g6g0", "g6g1", "g6g2", "g6g3", "g6g4", "g6g5", "g6g7", "g6g8",
    "g6g9", "g6h4", "g6h6", "g6h8", "g6i5", "g6i6", "g6i7", "g7a7", "g7b7", "g7c7", "g7d7", "g7e6", "g7e7", "g7e8", "g7f5",
    "g7f7", "g7f9", "g7g0", "g7g1", "g7g2", "g7g3", "g7g4", "g7g5", "g7g6", "g7g8", "g7g9", "g7h5", "g7h7", "g7h9", "g7i6",
    "g7i7", "g7i8", "g8a8", "g8b8", "g8c8", "g8d8", "g8e7", "g8e8", "g8e9", "g8f6", "g8f8", "g8g0", "g8g1", "g8g2", "g8g3",
    "g8g4", "g8g5", "g8g6", "g8g7", "g8g9", "g8h6", "g8h8", "g8i7", "g8i8", "g8i9", "g9a9", "g9b9", "g9c9", "g9d9", "g9e8",
    "g9e9", "g9f7", "g9f9", "g9g0", "g9g1", "g9g2", "g9g3", "g9g4", "g9g5", "g9g6", "g9g7", "g9g8", "g9h7", "g9h9", "g9i8",
    "g9i9", "h0a0", "h0b0", "h0c0", "h0d0", "h0e0", "h0f0", "h0f1", "h0g0", "h0g2", "h0h1", "h0h2", "h0h3", "h0h4", "h0h5",
    "h0h6", "h0h7", "h0h8", "h0h9", "h0i0", "h0i2", "h1a1", "h1b1", "h1c1", "h1d1", "h1e1", "h1f0", "h1f1", "h1f2", "h1g1",
    "h1g3", "h1h0", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "h1h9", "h1i1", "h1i3", "h2a2", "h2b2", "h2c2",
    "h2d2", "h2e2", "h2f1", "h2f2", "h2f3", "h2g0", "h2g2", "h2g4", "h2h0", "h2h1", "h2h3", "h2h4", "h2h5", "h2h6", "h2h7",
    "h2h8", "h2h9", "h2i0", "h2i2", "h2i4", "h3a3", "h3b3", "h3c3", "h3d3", "h3e3", "h3f2", "h3f3", "h3f4", "h3g1", "h3g3",
    "h3g5", "h3h0", "h3h1", "h3h2", "h3h4", "h3h5", "h3h6", "h3h7", "h3h8", "h3h9", "h3i1", "h3i3", "h3i5", "h4a4", "h4b4",
    "h4c4", "h4d4", "h4e4", "h4f3", "h4f4", "h4f5", "h4g2", "h4g4", "h4g6", "h4h0", "h4h1", "h4h2", "h4h3", "h4h5", "h4h6",
    "h4h7", "h4h8", "h4h9", "h4i2", "h4i4", "h4i6", "h5a5", "h5b5", "h5c5", "h5d5", "h5e5", "h5f4", "h5f5", "h5f6", "h5g3",
    "h5g5", "h5g7", "h5h0", "h5h1", "h5h2", "h5h3", "h5h4", "h5h6", "h5h7", "h5h8", "h5h9", "h5i3", "h5i5", "h5i7", "h6a6",
    "h6b6", "h6c6", "h6d6", "h6e6", "h6f5", "h6f6", "h6f7", "h6g4", "h6g6", "h6g8", "h6h0", "h6h1", "h6h2", "h6h3", "h6h4",
    "h6h5", "h6h7", "h6h8", "h6h9", "h6i4", "h6i6", "h6i8", "h7a7", "h7b7", "h7c7", "h7d7", "h7e7", "h7f6", "h7f7", "h7f8",
    "h7g5", "h7g7", "h7g9", "h7h0", "h7h1", "h7h2", "h7h3", "h7h4", "h7h5", "h7h6", "h7h8", "h7h9", "h7i5", "h7i7", "h7i9",
    "h8a8", "h8b8", "h8c8", "h8d8", "h8e8", "h8f7", "h8f8", "h8f9", "h8g6", "h8g8", "h8h0", "h8h1", "h8h2", "h8h3", "h8h4",
    "h8h5", "h8h6", "h8h7", "h8h9", "h8i6", "h8i8", "h9a9", "h9b9", "h9c9", "h9d9", "h9e9", "h9f8", "h9f9", "h9g7", "h9g9",
    "h9h0", "h9h1", "h9h2", "h9h3", "h9h4", "h9h5", "h9h6", "h9h7", "h9h8", "h9i7", "h9i9", "i0a0", "i0b0", "i0c0", "i0d0",
    "i0e0", "i0f0", "i0g0", "i0g1", "i0h0", "i0h2", "i0i1", "i0i2", "i0i3", "i0i4", "i0i5", "i0i6", "i0i7", "i0i8", "i0i9",
    "i1a1", "i1b1", "i1c1", "i1d1", "i1e1", "i1f1", "i1g0", "i1g1", "i1g2", "i1h1", "i1h3", "i1i0", "i1i2", "i1i3", "i1i4",
    "i1i5", "i1i6", "i1i7", "i1i8", "i1i9", "i2a2", "i2b2", "i2c2", "i2d2", "i2e2", "i2f2", "i2g0", "i2g1", "i2g2", "i2g3",
    "i2g4", "i2h0", "i2h2", "i2h4", "i2i0", "i2i1", "i2i3", "i2i4", "i2i5", "i2i6", "i2i7", "i2i8", "i2i9", "i3a3", "i3b3",
    "i3c3", "i3d3", "i3e3", "i3f3", "i3g2", "i3g3", "i3g4", "i3h1", "i3h3", "i3h5", "i3i0", "i3i1", "i3i2", "i3i4", "i3i5",
    "i3i6", "i3i7", "i3i8", "i3i9", "i4a4", "i4b4", "i4c4", "i4d4", "i4e4", "i4f4", "i4g3", "i4g4", "i4g5", "i4h2", "i4h4",
    "i4h6", "i4i0", "i4i1", "i4i2", "i4i3", "i4i5", "i4i6", "i4i7", "i4i8", "i4i9", "i5a5", "i5b5", "i5c5", "i5d5", "i5e5",
    "i5f5", "i5g4", "i5g5", "i5g6", "i5h3", "i5h5", "i5h7", "i5i0", "i5i1", "i5i2", "i5i3", "i5i4", "i5i6", "i5i7", "i5i8",
    "i5i9", "i6a6", "i6b6", "i6c6", "i6d6", "i6e6", "i6f6", "i6g5", "i6g6", "i6g7", "i6h4", "i6h6", "i6h8", "i6i0", "i6i1",
    "i6i2", "i6i3", "i6i4", "i6i5", "i6i7", "i6i8", "i6i9", "i7a7", "i7b7", "i7c7", "i7d7", "i7e7", "i7f7", "i7g6", "i7g7",
    "i7g8", "i7h5", "i7h7", "i7h9", "i7i0", "i7i1", "i7i2", "i7i3", "i7i4", "i7i5", "i7i6", "i7i8", "i7i9", "i8a8", "i8b8",
    "i8c8", "i8d8", "i8e8", "i8f8", "i8g7", "i8g8", "i8g9", "i8h6", "i8h8", "i8i0", "i8i1", "i8i2", "i8i3", "i8i4", "i8i5",
    "i8i6", "i8i7", "i8i9", "i9a9", "i9b9", "i9c9", "i9d9", "i9e9", "i9f9", "i9g8", "i9g9", "i9h7", "i9h9", "i9i0", "i9i1",
    "i9i2", "i9i3", "i9i4", "i9i5", "i9i6", "i9i7", "i9i8"};

std::vector<unsigned short> BuildMoveIndices() {
  std::vector<unsigned short> res(128 * 128);
  for (size_t i = 0; i < sizeof(kIdxToMove) / sizeof(kIdxToMove[0]); ++i) {
    res[kIdxToMove[i].as_packed_int()] = i;
  }
  return res;
}

const std::vector<unsigned short> kMoveToIdx = BuildMoveIndices();

BoardSquare Transform(BoardSquare sq, int transform) {
  if ((transform & FlipTransform) != 0) {
    sq.set(sq.row(), 8 - sq.col());
  }
  return sq;
}
}  // namespace

Move::Move(const std::string& str, bool black) {
  if (str.size() < 4) throw Exception("Bad move: " + str);
  SetFrom(BoardSquare(str.substr(0, 2), black));
  SetTo(BoardSquare(str.substr(2, 2), black));
  if (str.size() != 4) {
    throw Exception("Bad move: " + str);
  }
}

uint16_t Move::as_packed_int() const {
  return data_;
}

uint16_t Move::as_nn_index(int transform) const {
  if (transform == 0) {
    return kMoveToIdx[as_packed_int()];
  }
  Move transformed = *this;
  transformed.SetTo(Transform(to(), transform));
  transformed.SetFrom(Transform(from(), transform));
  return transformed.as_nn_index(0);
}

Move MoveFromNNIndex(int idx, int transform) {
  Move m = kIdxToMove[idx];
  if (transform == 0) {
    return m;
  }
  int inv_transform = transform;
  m.SetTo(Transform(m.to(), inv_transform));
  m.SetFrom(Transform(m.from(), inv_transform));
  return m;
}

}  // namespace lczero
