{-# LANGUAGE ScopedTypeVariables #-}
module GoogLeNet where
import           Control.Applicative
import           Control.Monad.State.Strict
import           Data.Graph.Inductive.Graph
import           Data.Graph.Inductive.PatriciaTree
import           Data.List
import           Data.Maybe
import           Data.Monoid
import           Debug.Trace
import           Graph
import           Layers

convolution :: Int -> Int -> Int -> Layer
convolution nOutput_ width_ stride_ = Convolution CP{
               nOutput=nOutput_,
               kernel=replicate 2 CD{width=width_, stride=stride_}}

inception :: GS (Node, Node)
inception = do
  splitId <- layer (Split 4)
  ids <- mapM buildColumn columns
  concatId <- layer (Concat 4)

  -- split joins at the bottom
  mapM_ (splitId >->) (map head ids)

  -- concat joins at the top
  mapM_ (>-> concatId) (map last ids)

  return (splitId, concatId)
      where
        buildColumn :: [Layer] -> GS [Int]
        buildColumn column_ = do
              layerIds <- mapM layer column_
              mapM_ (uncurry (>->)) (pairs layerIds)
              return layerIds
        n :: Int
        n = 4
        columns = [[convolution n 1 1],
                   [convolution n 1 1, convolution n 3 3],
                   [convolution n 1 1, convolution n 5 5],
                   [maxPool 2, convolution n 1 1]]


maxPool step = MaxPool MP{steps=replicate 2 step}
avgPool step = AveragePool MP{steps=replicate 2 step}

single :: GS Node -> GS (Node, Node)
single l = do
  x <- l
  return (x, x)

joined :: [GS (Node, Node)] -> GS (Node, Node)
joined columns = do
  bounds <- sequence columns
  go bounds
  traceShowM $ bounds
  -- mapM_ (uncurry (>->)) bounds
  return ((fst . head) bounds, (snd . last) bounds)
      where
        go [] = return ()
        go [x] = return ()
        go ((_, top):y@(bottom, _):zs) = do {top >-> bottom; go (y:zs)}




(>>->>) :: Node -> GS (Node, Node) -> GS Node
(>>->>) from column = do
  (bottom, top) <- column
  from >-> bottom
  return top

data GoogLeNet = GoogLeNet {input :: Node, midOutput :: Node, upperOutput :: Node, finalOutput :: Node}

googleNet = do
  input <- layer Input
  mid <- input >>->> joined [
          column initial,
          single (layer $ maxPool 3),
          inception,
          inception,
          single (layer $ maxPool 3),
          inception]
  midOutput <- mid >>->> column classifier
  upper <- mid >>->> joined [inception, inception, inception]

  -- upper classifier
  upperOutput <- upper >>->> column classifier
  final <- upper >>->> joined [inception, inception, inception]
  finalOutput <- final >>->> column [avgPool 7, fullyConnected 100 100, Criterion LogSoftMax]
  return $ GoogLeNet input midOutput upperOutput finalOutput
      where
        initial = [convolution 1 1 1, maxPool 2, Pointwise LocalResponseNormalize,
                   convolution 1 1 1, convolution 1 3 3, Pointwise LocalResponseNormalize]
        classifier = [avgPool 5, convolution 1 1 1,
              fullyConnected 100 100, fullyConnected 100 100,
              Criterion LogSoftMax]
