{-# LANGUAGE NamedFieldPuns      #-}
{-# LANGUAGE ScopedTypeVariables #-}
module GoogLeNet where
import           Control.Monad.State.Strict
import           Data.Graph.Inductive.Graph
import           Graph                      hiding (input)
import           Layers

single :: GS Node -> GS (Node, Node)
single l = do {x <- l; return (x, x)}

stack :: [GS (Node, Node)] -> GS (Node, Node)
stack columns = sequence columns >>= foldM1 merge
  where
    foldM1 _ [] = error "foldM1" "empty list"
    foldM1 f (x:xs) = foldM f x xs
    merge (bottom, midBelow) (midAbove, top) = midBelow >-> midAbove >> return (bottom, top)

(>>->>) :: Node -> GS (Node, Node) -> GS Node
from >>->> above = do
  (bottom, top) <- above
  from >-> bottom
  return top

inception :: GS (Node, Node)
inception = do
  splitId <- layer (Split 4)
  concatId <- layer (Concat 4)

  columnIds <- mapM buildColumn columns

  -- split joins at the bottom
  mapM_ ((splitId >->) . head) columnIds

  -- concat joins at the top
  mapM_ ((>-> concatId) . last) columnIds

  return (splitId, concatId)
      where
        buildColumn :: [Layer] -> GS [Int]
        buildColumn column_ = do
              layerIds <- mapM layer column_
              mapM_ (uncurry (>->)) (pairs layerIds)
              return layerIds
        columns = [[conv 4 1 1],
                   [conv 4 1 1, conv 4 3 3],
                   [conv 4 1 1, conv 4 5 5],
                   [pool Max 2, conv 4 1 1]]

data GoogLeNet = GoogLeNet {
      input        :: Node,
      middleOutput :: Node,
      upperOutput  :: Node,
      finalOutput  :: Node
    }

googleNet :: GS GoogLeNet
googleNet = do
  input <- layer Input

  middle <- input >>->> stack [
          column initial,
          single (layer $ pool Max 3),
          inception,
          inception,
          single (layer $ pool Max 3),
          inception]

  -- Middle classifier
  middleOutput <- middle >>->> column classifier

  upper <- middle >>->> stack [inception, inception, inception]

  -- upper classifier
  upperOutput <- upper >>->> column classifier
  finalOutput <- upper >>->> stack [inception, inception, inception, column finalClassifier]

  return GoogLeNet{input, middleOutput, upperOutput, finalOutput}
      where
        initial = [conv 1 1 1, pool Max 2, Pointwise LocalResponseNormalize,
                   conv 1 1 1, conv 1 3 3, Pointwise LocalResponseNormalize]
        classifier = [pool Avg 5, conv 1 1 1,
                      fullyConnected 100 100, fullyConnected 100 100,
                      Criterion LogSoftMax, Output]
        finalClassifier = [pool Avg 7, fullyConnected 100 100, Criterion LogSoftMax, Output]
