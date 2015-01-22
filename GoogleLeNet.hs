{-# LANGUAGE NamedFieldPuns      #-}
{-# LANGUAGE ScopedTypeVariables #-}
module GoogLeNet where
import           Control.Monad.State.Strict
import           Data.Graph.Inductive.Graph
import           Data.Monoid
import           Graph                      hiding (input)
import           Layers

convolution :: Int -> Int -> Int -> Layer
convolution nOutput_ width_ stride_ = Convolution CP{
               nOutput=nOutput_,
               kernel=replicate 2 CD{width=width_, stride=stride_}}


maxPool :: Int -> Layer
maxPool step = MaxPool MP{steps=replicate 2 step}

avgPool :: Int -> Layer
avgPool step = AveragePool MP{steps=replicate 2 step}

single :: GS Node -> GS (Node, Node)
single l = do {x <- l; return (x, x)}

newtype C = C { unC :: GS (Maybe (Node, Node)) }

instance Monoid C where
    mempty = C $ return Nothing
    C{unC=l} `mappend` C{unC=r} = C $ do {x <- l; y <- r; go x y}
        where
          go Nothing Nothing = return Nothing
          go (Just x) Nothing = return $ Just x
          go Nothing (Just y) = return $ Just y
          go (Just x) (Just y) = liftM Just (merge x y)

stack :: [GS (Node, Node)] -> GS (Node, Node)
stack columns = do
  let C{unC=result} = mconcat $ map (C . liftM Just) columns
  Just bounds <- result
  return bounds


merge :: (Node, Node) -> (Node, Node) -> GS (Node, Node)
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
        columns = [[convolution 4 1 1],
                   [convolution 4 1 1, convolution 4 3 3],
                   [convolution 4 1 1, convolution 4 5 5],
                   [maxPool 2, convolution 4 1 1]]

data GoogLeNet = GoogLeNet {input :: Node, middleOutput :: Node, upperOutput :: Node, finalOutput :: Node}

googleNet :: GS GoogLeNet
googleNet = do
  input <- layer Input

  middle <- input >>->> stack [
          column initial,
          single (layer $ maxPool 3),
          inception,
          inception,
          single (layer $ maxPool 3),
          inception]

  -- Middle classifier
  middleOutput <- middle >>->> column classifier
  upper <- middle >>->> stack [inception, inception, inception]

  -- upper classifier
  upperOutput <- upper >>->> column classifier
  finalOutput <- upper >>->> stack [inception, inception, inception, column finalClassifier]

  return GoogLeNet{input, middleOutput, upperOutput, finalOutput}
      where
        initial = [convolution 1 1 1, maxPool 2, Pointwise LocalResponseNormalize,
                   convolution 1 1 1, convolution 1 3 3, Pointwise LocalResponseNormalize]
        classifier = [avgPool 5, convolution 1 1 1,
                      fullyConnected 100 100, fullyConnected 100 100,
                      Criterion LogSoftMax, Output]
        finalClassifier = [avgPool 7, fullyConnected 100 100, Criterion LogSoftMax, Output]
