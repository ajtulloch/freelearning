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
  (classifierId, _) <- column classifier

  -- Connect the join of the features to the classifier
  connect joinId classifierId
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

updateActivation :: G b -> (Layer -> [a] -> a) -> Node -> State (EdgeMap a) ()
updateActivation graph f gid = do
  let inputIds = pre graph gid
  let l = fromJust (lab graph gid)
  acts <- get
  let inputActs = map (\k -> fromJust (HM.lookup (k, gid) acts)) inputIds
  let outputAct = f l inputActs
  put (setOutgoingActivations graph gid outputAct acts)

setOutgoingActivations :: G b -> Node -> a -> EdgeMap a -> EdgeMap a
setOutgoingActivations graph gid outputAct acts = foldl step acts outputIds where
    outputIds = suc graph gid
    step m outputId = HM.insert (gid, outputId) outputAct m

updateEdges :: G b -> EdgeMap a -> G a
updateEdges graph acts = gmap update graph
    where
      update (pres, gid, l, sucs) = (newpre, gid, l, newsucc)
          where
            newpre = map (\(_, preId) -> (actAt acts preId gid, preId)) pres
            newsucc = map (\(_, succId) -> (actAt acts gid succId, succId)) sucs

actAt :: EdgeMap a -> Node -> Node -> a
actAt acts from to = fromJust (HM.lookup (from, to) acts)

activations :: G b -> (Layer -> [a] -> a) -> a -> EdgeMap a
activations graph f input = execState (mapM step layers) initialState
    where
      (inputLayer:layers) = topsort graph
      step = updateActivation graph f
      initialState = setOutgoingActivations graph inputLayer input HM.empty

fpropAll :: Layer -> [[Float]] -> [Float]
fpropAll l inputs = fprop l (concat inputs)

sizeAll :: Layer -> [[Int]] -> [Int]
sizeAll l inputs = trace (printf "L: %s, I: %s" (show (P l)) (show inputs)) $
                   outputSize l $ go inputs
    where
      go [] = []
      go (x:_) = x

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

visualize :: G a ->  b -> (Layer -> [b] -> b) -> (b -> String) -> IO ()
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
