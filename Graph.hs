module Graph where

import           Control.Monad.State.Strict

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

type G = Gr Layer
type GS = State (Int, G ())

layer :: Layer -> GS Int
layer l = do
  (gid, s) <- get
  put (gid + 1, insNode (gid, l) s)
  return gid

connect :: Int -> Int -> GS ()
connect from to = modify (second (insEdge (from, to, ())))

column :: [Layer] -> GS (Int, Int)
column layers = do
  ids <- mapM layer layers
  mapM_ (uncurry connect) (pairs ids)

  return (head ids, last ids)
    where
      pairs [] = []
      pairs xs = zip (init xs) (tail xs)

fork :: [[Layer]] -> GS (Int, Int)
fork columns = do
  splitId <- layer $ ModelParallelFork (length columns)

  ids <- mapM column columns
  mapM_ (connect splitId . fst) ids

  joinId <- layer $ ModelParallelJoin (length columns)
  mapM_ ((`connect` joinId) . snd) ids

  return (splitId, joinId)

-- lstm :: GS ()
-- lstm = do
--   return ()
--   lstmInput <-
--   inputId <- layer (LSTM Input)
--   outputId <- layer (LSTM Output)



-- | DSL Example
alexNet :: G ()
alexNet = toGraph $ do
  -- Create the input layer
  inputId <- layer Input

  -- Feature mapping, split across feature maps
  (splitId, joinId) <- fork (replicate 2 features)

  -- Connect the input to the start of the split
  connect inputId splitId

  -- Create the classifier
  (classifierStart, classifierEnd) <- column classifier

  -- Connect the join of the features to the classifier
  connect joinId classifierStart

  outputId <- layer Output
  connect classifierEnd outputId

    where
      features = [
           Convolution CP{nOutput=96, kernel=replicate 2 CD{width=11, stride=4}},
           Pointwise ReLU,
           MaxPool MP{steps=[2, 2]},
           -- Convolution, Pointwise ReLU, MaxPool,
           -- Convolution, Pointwise ReLU,
           -- Convolution, Pointwise ReLU,
           Convolution CP{nOutput=1024, kernel=replicate 2 CD{width=3, stride=1}},
           Pointwise ReLU, MaxPool MP{steps=[2, 2]}]

      classifier = [
           Reshape,
           Pointwise DropOut,
           FullyConnected FC{nHidden=3072, weights=[[]]},
           Pointwise ReLU,
           Pointwise DropOut,
           FullyConnected FC{nHidden=4096, weights=[[]]},
           Pointwise ReLU,
           FullyConnected FC{nHidden=1000, weights=[[]]},
           Criterion LogSoftMax
          ]


toGraph :: GS a -> G ()
toGraph f = snd (execState f (0, empty))


type EdgeMap a = HM.HashMap (Node, Node) a

initializeMap :: G b -> Node -> a -> EdgeMap a
initializeMap graph root output = foldl step HM.empty (suc graph root)
  where
    step m outputId = HM.insert (root, outputId) output m

updateActivation :: (Show a) => G b -> (Layer -> [a] -> [a]) -> Node -> State (EdgeMap a) ()
updateActivation graph f gid = do
  let inputIds = pre graph gid
  let l = fromJust (lab graph gid)
  acts <- get
  let inputActs = map (\k -> fromJust (HM.lookup (k, gid) acts)) inputIds

  let outputIds = suc graph gid
  let outputActs = f l inputActs
  when (length outputIds /= length outputActs) $
       case l of
         Output -> return ()
         _ -> error $ printf "Mismatched output: %s, %s, %s" (show l) (show outputIds) (show outputActs)

  let updated = foldl step acts (zip outputIds outputActs)
  put updated
    where
      step m (outputId, outputAct) = HM.insert (gid, outputId) outputAct m


updateEdges :: G b -> EdgeMap a -> G a
updateEdges graph acts = gmap update graph
    where
      update (pres, gid, l, sucs) = (newpre, gid, l, newsucc)
          where
            newpre = map (\(_, preId) -> (actAt acts preId gid, preId)) pres
            newsucc = map (\(_, succId) -> (actAt acts gid succId, succId)) sucs

actAt :: EdgeMap a -> Node -> Node -> a
actAt acts from to = fromJust (HM.lookup (from, to) acts)

activations :: (Show a) => G b -> (Layer -> [a] -> [a]) -> a -> EdgeMap a
activations graph f input = execState (mapM step layers) (initializeMap graph root input)
    where
      (root:layers) = topsort graph
      step = updateActivation graph f

fpropAll :: Layer -> [Tensor] -> [Tensor]
fpropAll l inputs = map (fprop l) inputs


sizeAll :: Layer -> [[Int]] -> [[Int]]
sizeAll l@(ModelParallelFork n) input = (replicate n) $ outputSize l (head input)
sizeAll l@(ModelParallelJoin _) input = [outputSize l (head input)]
sizeAll l inputs = trace (printf "L: %s, I: %s" (show (P l)) (show inputs)) $  map (outputSize l) inputs

fullyConnected :: Int -> Int -> Layer
fullyConnected nHidden_ nInput =
    FullyConnected FC{nHidden=nHidden_, weights=[[1 | _ <- [1..nInput]] | _ <- [1..nHidden_]]}

simpleNet :: G ()
simpleNet = toGraph $
            column [Input,
                    fullyConnected 2 2,
                    Pointwise ReLU,
                    fullyConnected 2 2,
                    Pointwise ReLU,
                    Criterion LogSoftMax]

visualize :: (Show a, Show b) => G a ->  b -> (Layer -> [b] -> [b]) -> (b -> String) -> IO ()
visualize graph initial propFn printFn = do
  let edgeActivatedGraph = updateEdges graph (activations graph propFn initial)
  let dot = (showDot . fglToDotString . emap printFn . nmap (show . P)) edgeActivatedGraph
  writeFile "file.dot" dot
  void $ system "dot -Tpng -ofile.png file.dot"
  void $ system "open file.png &"
  return ()

alexNetActivations :: IO ()
alexNetActivations = visualize alexNet [1.0, 2.0] fpropAll (printf "%s" . show)

alexNetCommunication :: IO ()
alexNetCommunication = visualize alexNet [4, 256, 256] sizeAll (\e -> printf "%s (%s)" (show (product e)) (intercalate "x" (map show e)))
