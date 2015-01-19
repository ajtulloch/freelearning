module Data.Graph.Free (freeGraph, freeFoldableGraph, formatted, generate) where

import           Control.Monad.Free
import           Control.Monad.State
import           Data.Foldable
import           Data.Graph.Cofree
import           Data.Graph.Inductive.Graph
import           Data.Graph.Inductive.Tree
import           Data.Graph.Internal
import           Data.GraphViz

freeFoldableGraph :: Foldable f => Free f a -> Gr (Free f a) ()
freeFoldableGraph = freeGraph toList

freeGraph :: (f (Free f a) -> [Free f a]) -> Free f a -> Gr (Free f a) ()
freeGraph = runGraphState freeGraphState

freeGraphState :: Node -> (f (Free f a) -> [Free f a]) -> Free f a -> State Node ([LNode (Free f a)], [UEdge])
freeGraphState = recursiveGraphState unwrap
    where unwrap (Pure _) = Nothing
          unwrap (Free f) = Just f

generate :: Gr f () -> (f -> Attributes) -> String -> IO ()
generate g a n = fmap (const ()) . runGraphviz (graphToDot (formatted a defaultParams) g) Png $ n ++ ".png"
