module CUDA where

import           Foreign.CUDA
import           Foreign.CUDA.Cublas
import           Foreign.CUDA.Driver.Device

x = do
  h <- Foreign.CUDA.count
  print h
  initialise []
  handle <- create
  destroy handle


