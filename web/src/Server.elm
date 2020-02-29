port module Server exposing (..)

import Json.Encode exposing ( encode, object, string, bool, float, int )

port toServer : String -> Cmd msg
port toClient : (String -> msg) -> Sub msg

type CostCombineFunction =
  CombineMin |
  CombineSum

type alias SimilarityMeasure = {
  name : String,
  quantiles : Bool}

type alias QuerySettings = {
  ignoreDeterminers : Bool,
  posWeighting : Float,
  posMismatch : Float,
  mismatchLengthPenalty : Int,
  submatchWeight : Float,
  idfWeight : Float,
  bidirectional : Bool,
  similarityThreshold : Float,
  similarityFalloff : Float,
  similarityMeasure : SimilarityMeasure,
  mixEmbedding : Float,
  enableElmo : Bool,
  annotatePOS : Bool,
  annotateDebug : Bool,
  costCombine : CostCombineFunction}

similarityMeasureInternalId : SimilarityMeasure -> String
similarityMeasureInternalId m =
  if m.quantiles then ("ranked-" ++ m.name) else m.name

startSearch : String -> QuerySettings -> Cmd msg
startSearch query settings = toServer (encode 0 (object
  [ ("command", string "start-search")
  , ("query", string query)
  , ("ignore_determiners", bool settings.ignoreDeterminers)
  , ("pos_weighting", float settings.posWeighting)
  , ("pos_mismatch", float settings.posMismatch)
  , ("mismatch_length_penalty", int settings.mismatchLengthPenalty)
  , ("submatch_weight", float settings.submatchWeight)
  , ("idf_weight", float settings.idfWeight )
  , ("bidirectional", bool settings.bidirectional)
  , ("similarity_threshold", float settings.similarityThreshold)
  , ("similarity_falloff", float settings.similarityFalloff)
  , ("similarity_measure", string (similarityMeasureInternalId settings.similarityMeasure))
  , ("cost_combine_function", string (if settings.costCombine == CombineMin then "min" else "sum"))
  , ("mix_embedding", float settings.mixEmbedding)
  , ("enable_elmo", bool settings.enableElmo)
  ]))

abortSearch : Cmd msg
abortSearch = toServer (encode 0 (object
  [ ("command", string "abort-search")
  ]))
