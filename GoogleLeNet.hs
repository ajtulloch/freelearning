{-# LANGUAGE NamedFieldPuns #-}
module GoogLeNet where

import           Data.Graph.Inductive.Graph
import           Graph                      hiding (input)
import           Layers
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

data GoogLeNet = GoogLeNet { input, middleOutput, upperOutput, finalOutput :: Node }

googleNet :: GS GoogLeNet
googleNet = do
  input <- layer Input
  middle <- return input
           >>->> column initial
           >>->> (single . layer) (pool Max 3)
           >>->> inception
           >>->> inception
           >>->> (single . layer) (pool Max 3)
           >>->> inception

  -- Middle classifier
  middleOutput <- return middle
                 >>->> column classifier

  upper <- return middle
          >>->> inception
          >>->> inception
          >>->> inception
          >>->> inception

  -- upper classifier
  upperOutput <- return upper
                >>->> column classifier

  finalOutput <- return upper
                >>->> inception
                >>->> inception
                >>->> inception
                >>->> column finalClassifier

  return GoogLeNet{input, middleOutput, upperOutput, finalOutput}
      where
        initial = [conv 1 1 1, pool Max 2, Pointwise LocalResponseNormalize,
                   conv 1 1 1, conv 1 3 3, Pointwise LocalResponseNormalize]
        classifier = [pool Avg 5, conv 1 1 1,
                      fullyConnected 100 100, fullyConnected 100 100,
                      Criterion LogSoftMax, Output]
        finalClassifier = [pool Avg 7, fullyConnected 100 100, Criterion LogSoftMax, Output]
