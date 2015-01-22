{-# LANGUAGE RecordWildCards #-}
module Graph where

import           Control.Monad.State.Strict

import           Control.Applicative               hiding (empty)
import           Control.Arrow
import           Data.Graph.Inductive.Dot
import           Data.Graph.Inductive.Graph
import           Data.Graph.Inductive.PatriciaTree
import           Data.Graph.Inductive.Query
import qualified Data.HashMap.Strict               as HM
import           Data.List
import           Data.Maybe
import           Debug.Trace
import           Layers
import           System.Process
import           Text.Printf

type G a = State (Int, Gr a ())
type GS = G Layer

layer :: a -> State (Node, Gr a b) Node
layer l = do
  (gid, s) <- get
  put (gid + 1, insNode (gid, l) s)
  return gid

(>->) :: Int -> Int -> G a ()
(>->) from to = modify (second (insEdge (from, to, ())))

column :: [a] -> G a (Node, Node)
column = stack . map (single . layer)

single :: G a Node -> G a (Node, Node)
single l = do {x <- l; return (x, x)}

stack :: [G a (Node, Node)] -> G a (Node, Node)
stack columns = sequence columns >>= foldM1 merge
  where
    foldM1 _ [] = error "foldM1" "empty list"
    foldM1 f (x:xs) = foldM f x xs
    merge (bottom, midBelow) (midAbove, top) = midBelow >-> midAbove >> return (bottom, top)

(>>->>) :: Node -> G a (Node, Node) -> G a Node
from >>->> above = do
  (bottom, top) <- above
  from >-> bottom
  return top


pairs :: [b] -> [(b, b)]
pairs [] = []
pairs xs = zip xs (tail xs)

fork :: [[Layer]] -> G Layer (Int, Int)
fork columns = do
  splitId <- layer $ ModelParallelFork (length columns)

  ids <- mapM column columns
  mapM_ ((splitId >->) . fst) ids

  joinId <- layer $ ModelParallelJoin (length columns)
  mapM_ ((>-> joinId) . snd) ids

  return (splitId, joinId)

data LSTMG = LSTMG {input  :: Node, output :: Node, cell :: Node}

lstm :: GS LSTMG
lstm = do
  config@LSTMG{..} <- LSTMG <$> go LSInput <*> go LSOutput <*> go LSCell

  input >-> cell
  cell >-> output

  return config
    where
       go = layer . LSTM

lstmUnfolded :: Int -> GS [LSTMG]
lstmUnfolded n = do
  lstms <- replicateM n lstm
  mapM_ (\(prev, next) -> cell prev >-> cell next) (pairs lstms)
  return lstms


viewBuilder :: GS a -> IO ()
viewBuilder = view . toGraph

view ::  Gr Layer b -> IO ()
view g = do
  let dot = (showDot . fglToDotString . emap (const "") . nmap (show . P)) g
  printDot dot

-- | DSL Example
alexNetG :: Gr Layer ()
alexNetG = toGraph alexNet

alexNet :: GS (Int, Int)
alexNet = do
  -- Create the input layer
  inputId <- layer Input

  -- Feature mapping, split across feature maps
  joinId <- inputId >>->> fork (replicate 2 features)

  -- Create the classifier
  outputId <- joinId >>->> column classifier

  return (inputId, outputId)
    where
      features = [
           Convolution CP{nOutput=96, kernel=replicate 2 CD{width=11, stride=4}},
           Pointwise ReLU,
           Pool Max PP{steps=[2, 2]},
           -- Convolution, Pointwise ReLU, MaxPool,
           -- Convolution, Pointwise ReLU,
           -- Convolution, Pointwise ReLU,
           Convolution CP{nOutput=1024, kernel=replicate 2 CD{width=3, stride=1}},
           Pointwise ReLU, Pool Max PP{steps=[2, 2]}]

      classifier = [
           Reshape,
           Pointwise DropOut,
           FullyConnected FC{nHidden=3072, weights=[[]]},
           Pointwise ReLU,
           Pointwise DropOut,
           FullyConnected FC{nHidden=4096, weights=[[]]},
           Pointwise ReLU,
           FullyConnected FC{nHidden=1000, weights=[[]]},
           Criterion LogSoftMax,
           Output
          ]


toGraph :: G a b -> Gr a ()
toGraph f = snd (execState f (0, empty))


type EdgeMap a = HM.HashMap (Node, Node) a

initializeMap  :: Graph gr => gr a b -> Node -> v -> EdgeMap v
initializeMap graph root output = foldl step HM.empty (suc graph root)
  where
    step m outputId = HM.insert (root, outputId) output m

updateActivation :: (Graph gr) => gr a b -> (a -> [v] -> [v]) -> Node -> State (EdgeMap v) ()
updateActivation graph f gid = do
  let inputIds = pre graph gid
  let l = fromJust (lab graph gid)
  acts <- get
  let inputActs = map (\k -> fromJust (HM.lookup (k, gid) acts)) inputIds

  let outputIds = suc graph gid
  let outputActs = f l inputActs
  when (not (null outputIds) && length outputIds /= length outputActs) $ error "Mismatch"

  let updated = foldl step acts (zip outputIds outputActs)
  put updated
    where
      step m (outputId, outputAct) = HM.insert (gid, outputId) outputAct m


updateEdges :: DynGraph gr => gr c b -> EdgeMap d -> gr c d
updateEdges graph acts = gmap update graph
    where
      update (pres, gid, l, sucs) = (newpre, gid, l, newsucc)
          where
            newpre = map (\(_, preId) -> (actAt acts preId gid, preId)) pres
            newsucc = map (\(_, succId) -> (actAt acts gid succId, succId)) sucs

actAt :: EdgeMap a -> Node -> Node -> a
actAt acts from to = fromJust (HM.lookup (from, to) acts)

activations :: Graph gr => gr a b -> (a -> [v] -> [v]) -> v -> EdgeMap v
activations graph f input = execState (mapM step layers) (initializeMap graph root input)
    where
      (root:layers) = topsort graph
      step = updateActivation graph f

fpropAll :: Layer -> [Tensor] -> [Tensor]
fpropAll l = map (fprop l)

sizeAll :: Layer -> [[Int]] -> [[Int]]
sizeAll l@(ModelParallelFork n) input = replicate n $ outputSize l (head input)
sizeAll l@(ModelParallelJoin _) input = [outputSize l (head input)]
sizeAll l inputs = trace (printf "L: %s, I: %s" (show (P l)) (show inputs)) $  map (outputSize l) inputs

fullyConnected :: Int -> Int -> Layer
fullyConnected nHidden_ nInput =
    FullyConnected FC{nHidden=nHidden_, weights=[[1 | _ <- [1..nInput]] | _ <- [1..nHidden_]]}

simpleNet :: Gr Layer ()
simpleNet = toGraph $
            column [Input,
                    fullyConnected 2 2,
                    Pointwise ReLU,
                    fullyConnected 2 2,
                    Pointwise ReLU,
                    Criterion LogSoftMax]

visualize :: Gr Layer a -> b -> (Layer -> [b] -> [b]) -> (b -> String) -> IO ()
visualize graph initial propFn printFn = do
  let edgeActivatedGraph = updateEdges graph (activations graph propFn initial)
  let dot = (showDot . fglToDotString . emap printFn . nmap (show . P)) edgeActivatedGraph
  printDot dot

printDot :: String -> IO ()
printDot dot = do
  writeFile "file.dot" dot
  void $ system "dot -Tpng -ofile.png file.dot"
  void $ system "open file.png &"

alexNetActivations :: IO ()
alexNetActivations = visualize alexNetG [1.0, 2.0] fpropAll (printf "%s" . show)

alexNetCommunication :: IO ()
alexNetCommunication = visualize alexNetG [4, 256, 256] sizeAll (\e -> printf "%s (%s)" (show (product e)) (intercalate "x" (map show e)))
