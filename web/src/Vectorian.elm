module Vectorian exposing (..)

import Server exposing (..)
import Msg exposing (..)
import MismatchPenalty exposing (..)

import Debug
import Browser
import Html exposing (Html, main_, text, div, span, small, p, br, Attribute, img)
import Html.Events exposing (on, keyCode, onClick, onInput, onCheck )
import Html.Attributes exposing (placeholder, style, class, attribute, src, width)

import Bulma.Modifiers exposing (..)
import Bulma.Modifiers.Typography exposing (
  italicize, textWeight, textColor, textSize, Weight(..), Color(..), Size(..), textCentered )
import Bulma.Elements exposing (..)
import Bulma.Layout exposing (..)
import Bulma.Form exposing (..)
import Bulma.Components exposing (..)

import BulmaExtensions exposing (
  checkbox, slider, radio, fineGrainedSlider,
  mismatchCutoffSlider, falloffSlider, optionalCheckbox, intSlider )


import Json.Decode as Decode exposing (int, list, string, float, bool, Decoder)
import Json.Decode.Pipeline exposing (required, optional)
import Json.Decode as Json

import Round

type SearchStatus =
  NotSearching |
  SearchRequested |
  Searching (Maybe Float) |
  AbortRequested

type alias MatchLocation = {
  speaker : String,
  author : String,
  work : String,
  location : String }

type alias MatchDebugInfo = {
  document : Int,
  sentence : Int }

type alias Match = {
  debug : MatchDebugInfo,
  score : Float,
  algorithm : String,
  location : MatchLocation,
  regions : List Region,
  omitted : List String }

type alias Region = {
  s : String,
  mismatch_penalty : Float,
  t : String, -- matched query word, might be empty for unmatched corpus insertions
  similarity : Float,
  weight : Float,
  pos_s : String,
  pos_t : String,
  metric : String }

regionDecoder : Decoder Region
regionDecoder =
  Decode.succeed Region
    |> required "s" string
    |> optional "mismatch_penalty" float 0.0
    |> optional "t" string ""
    |> optional "similarity" float 0.0
    |> optional "weight" float 0.0
    |> optional "pos_s" string ""
    |> optional "pos_t" string ""
    |> optional "metric" string ""

matchLocationDecoder : Decoder MatchLocation
matchLocationDecoder =
  Decode.succeed MatchLocation
    |> required "speaker" string
    |> required "author" string
    |> required "work" string
    |> required "location" string

matchDebugInfoDecoder : Decoder MatchDebugInfo
matchDebugInfoDecoder =
  Decode.succeed MatchDebugInfo
    |> required "document" int
    |> required "sentence" int

matchDecoder : Decoder Match
matchDecoder =
  Decode.succeed Match
    |> required "debug" matchDebugInfoDecoder
    |> required "score" float
    |> required "algorithm" string
    |> required "location" matchLocationDecoder
    |> required "regions" (list regionDecoder)
    |> optional "omitted" (list string) []

type alias ServerResultsMessage = {
  command: String, results : List Match, progress : Float }

serverResultsMessageDecoder : Decoder ServerResultsMessage
serverResultsMessageDecoder =
  Decode.succeed ServerResultsMessage
    |> required "command" string
    |> required "results" (list matchDecoder)
    |> required "progress" float


type alias Features = {
  nicdm : Bool,
  apsynp : Bool,
  maximum : Bool,
  quantiles : Bool,
  idf : Bool}

type alias ServerConnectedMessage = {
  status: String, features: Features }

featuresDecoder : Decoder Features
featuresDecoder =
  Decode.succeed Features
    |> required "nicdm" bool
    |> required "apsynp" bool
    |> required "maximum" bool
    |> required "quantiles" bool
    |> required "idf" bool

serverConnectedMessageDecoder : Decoder ServerConnectedMessage
serverConnectedMessageDecoder =
  Decode.succeed ServerConnectedMessage
    |> required "status" string
    |> required "features" featuresDecoder


defaultQuerySettings : QuerySettings
defaultQuerySettings = {
  ignoreDeterminers = False,
  posWeighting = 50,
  posMismatch = 50,
  mismatchLengthPenalty = 5,
  submatchWeight = 0,
  idfWeight = 0,
  bidirectional = False,
  similarityFalloff = 1,
  similarityThreshold = 10,
  similarityMeasure = {name = "cosine", quantiles = False},
  costCombine = CombineSum,
  mixEmbedding = 0,
  enableElmo = False,
  annotatePOS = False,
  annotateDebug = False}

type alias Model = {
  connected : Bool,
  features : Features,
  query : String,
  querySettings : QuerySettings,
  search : SearchStatus,
  results : List Match }

init : (Model, Cmd Msg)
init = ( {
  connected = True,  -- binding js ensures we're connected
  -- at first time of instantation of elm app.
  features = {nicdm = True, apsynp = True, maximum = False, quantiles = False, idf = True},
  query = "",
  querySettings = defaultQuerySettings,
  search = NotSearching,
  results = [] }, Cmd.none )

subscriptions : Model -> Sub Msg
subscriptions model = Server.toClient ReceiveDataFromServer

startSearch : Model -> (Model, Cmd Msg)
startSearch model =
  (
    { model | search = SearchRequested, results = [] }
    , Server.startSearch model.query model.querySettings
  )

sortResults : List Match -> List Match
sortResults results =
   List.reverse (List.sortBy .score results)

updateQuerySettings : QuerySettings -> QuerySettingsMsg -> QuerySettings
updateQuerySettings settings msg =
  case msg of
    IgnoreDeterminers on -> { settings | ignoreDeterminers = on }
    EnableElmo on -> { settings | enableElmo = on }
    AnnotatePOS on -> { settings | annotatePOS = on }
    AnnotateDebug on -> { settings | annotateDebug = on }
    PosWeighting m ->
      case m of
        Just x -> { settings | posWeighting = x }
        Nothing -> settings
    PosMismatch m ->
      case m of
        Just x -> { settings | posMismatch = x }
        Nothing -> settings
    MismatchLengthPenalty m ->
      case m of
        Just x -> { settings | mismatchLengthPenalty = x }
        Nothing -> settings
    SubmatchWeight m ->
      case m of
        Just x -> { settings | submatchWeight = x }
        Nothing -> settings
    IDFWeight m ->
      case m of
        Just x -> { settings | idfWeight = x }
        Nothing -> settings
    Bidirectional on -> { settings | bidirectional = on }
    SimilarityFalloff m ->
      case m of
        Just x -> { settings | similarityFalloff = x }
        Nothing -> settings
    SimilarityThreshold m ->
      case m of
        Just x -> { settings | similarityThreshold = x }
        Nothing -> settings
    UpdateSimilarityMeasure m ->
      case m of
        Just x -> { settings | similarityMeasure = x }
        Nothing -> settings
    CostCombine c -> { settings | costCombine = c }
    MixEmbedding m ->
      case m of
        Just x -> { settings | mixEmbedding = x }
        Nothing -> settings

searching : Float -> SearchStatus
searching p =
  if p > 0 then Searching (Just p) else Searching Nothing

update : Msg -> Model -> (Model, Cmd Msg)
update msg model =
    case msg of
      NoOp -> (model, Cmd.none)

      QueryKeyDown key ->
        if key == 13 then
          startSearch model
        else
          (model, Cmd.none)
      StartSearch ->
        startSearch model
      AbortSearch ->
        ({ model | search = AbortRequested }, Server.abortSearch)

      QueryInput s ->
        ({model | query = s }, Cmd.none)
      UpdateQuerySettings q ->
        ({model | querySettings = updateQuerySettings model.querySettings q }, Cmd.none)

      ReceiveDataFromServer "connected" ->
        ({ model | connected = True }, Cmd.none)
      ReceiveDataFromServer "disconnected" ->
        ({ model | connected = False }, Cmd.none)
      ReceiveDataFromServer "search-started" ->
        ({ model | search = Searching Nothing }, Cmd.none)
      ReceiveDataFromServer "search-aborted" ->
        ({ model | search = NotSearching }, Cmd.none)
      ReceiveDataFromServer "search-finished" ->
        ({ model | search = NotSearching }, Cmd.none)
      ReceiveDataFromServer message ->
        let
          decoded = Decode.decodeString serverResultsMessageDecoder message
          r = model.results
        in
          case decoded of
            Ok serverMessage ->
              if serverMessage.command == "add-results" then
                ({ model |
                  results = sortResults(r ++ serverMessage.results)
                  , search = searching serverMessage.progress
                }, Cmd.none)
              else
                (model, Cmd.none)
            Err err -> (model, Cmd.none)

onKeyDown : (Int -> msg) -> Attribute msg
onKeyDown tagger =
  on "keydown" (Decode.map tagger keyCode)

main : Program () Model Msg
main
  = Browser.element
    { init = \() -> init
    , subscriptions = subscriptions
    , view = view
    , update = update
    }

view : Model -> Html Msg
view model
  = main_ []
    (
      if model.connected then
      [
        searchUI model
      ]
      else
      [
        section NotSpaced []
        [
          notification Bulma.Modifiers.Danger []
          [
            text "Lost server connection. Please try again later."
          ]
        ]
      ]
    )

joinRegions : Region -> List Region -> (String -> String -> String) -> List Region
joinRegions left rest joinStr
  = case List.head rest of
    Just h ->
      if h.similarity * h.weight == 0 then
        trimRegionsHead ([{
          s = (joinStr left.s  h.s), mismatch_penalty = 0, t = "", similarity = 0, weight = 0,
          pos_s = "", pos_t = "", metric = ""}] ++ (List.drop 1 rest)) joinStr
      else [left] ++ rest
    Nothing -> [left] ++ rest

trimRegionsHead : List Region -> (String -> String -> String) -> List Region
trimRegionsHead regions joinStr
  = case List.head regions of
    Just r ->
      if r.similarity * r.weight == 0
      then joinRegions r (List.drop 1 regions) joinStr
      else regions
    Nothing -> regions

trimRegions : List Region -> List Region
trimRegions r
  = List.reverse (trimRegionsHead (List.reverse (trimRegionsHead r (\a b -> a ++ b))) (\a b -> b ++ a))

scoreColor : Float -> Bulma.Modifiers.Color
scoreColor score
  = if score <= 0.75 then Bulma.Modifiers.Warning
    else if score <= 0.25 then Bulma.Modifiers.Danger
    else Bulma.Modifiers.Success

opacity : Region -> String
opacity region
  = "opacity: " ++ String.fromFloat region.weight ++ ";"

matchRegionView : Model -> Region -> Html Msg
matchRegionView model region
  =
  let
    s_html =
      span
      [ textWeight Bold
      , textColor Bulma.Modifiers.Typography.Black
      ] [ text region.s ]
    t_html =
      tag { tagModifiers | color = Bulma.Modifiers.Light } [ ] [ text region.t ]
  in
    span []
    [
      span [ attribute "style" "display:inline-table" ]
        [ span [ attribute "style" "display:table-row" ]
          [ span [ attribute "style" "display:table-cell" ] [
            s_html
            ]
          , span [ attribute "style" "display:table-cell" ] [
            t_html
            ]
          , span [ attribute "style" ("display:table-cell;" ++ (opacity region)) ] [
            text " "
            , tag { tagModifiers | color = scoreColor region.similarity }
            [ ] [ text (String.fromInt (floor (100 * region.similarity)) ++ "%") ]
          ]
        ]
        , if model.querySettings.annotatePOS then
          span [ attribute "style" "display:table-row" ] [
            span
              [
                attribute "style" "display:table-cell; padding-left: 0.2em; padding-right: 0.2em;"
                , textSize Bulma.Modifiers.Typography.Small
                , textCentered
                , if region.pos_s == region.pos_t
                  then textColor Bulma.Modifiers.Typography.Black
                  else textColor Bulma.Modifiers.Typography.Danger
              ] [
                text region.pos_s
              ]
            , span
              [
                attribute "style" "display:table-cell; padding-left: 0.2em; padding-right: 0.2em;"
                , textSize Bulma.Modifiers.Typography.Small
                , textCentered
              ] [
                text region.pos_t
              ]
            , span
              [ attribute "style" "display:table-cell; padding-left: 0.2em; padding-right: 0.2em;"
                , textSize Bulma.Modifiers.Typography.Small
                , textCentered
                , textColor GreyLight
              ] [
                text region.metric
              ]
          ]
        else
          span [] []
      ]
    ]

regionView : Model -> Region -> Html Msg
regionView model region
  = if String.length region.t > 0 && region.similarity * region.weight > 0
    then matchRegionView model region
    else
      if model.querySettings.annotatePOS && region.mismatch_penalty > 0
      then
        span []
        [
          span [ attribute "style" "display:inline-table" ]
          [
            span [ attribute "style" "display:table-row", textColor GreyLight ]
            [
              text region.s
            ]
            , span [ attribute "style" "display:table-row;", textCentered ]
            [
              tag { tagModifiers | color = Bulma.Modifiers.Danger }
              [ ] [ text ("-" ++ (Round.round 1 (100 * region.mismatch_penalty)) ++ "% rel") ]
            ]
          ]
        ]
      else
        span [ textColor GreyLight ] [ text region.s ]

matchScoreView : Match -> Html Msg
matchScoreView match
  = span [ textWeight Bold ]
  [ text (Round.round 1 (100 * match.score) ++ "%") ]

matchView : Model -> Match -> Html Msg
matchView model match
  = media []
    [
      mediaLeft []
      [
        p [ class "image is-64x64" ] [
          span [ class "buttons" ]
          [
            matchScoreView match

            , br [] []
            , div []

              (if List.length match.omitted <= 2 then
                (List.map (\x -> div [attribute "style" "text-decoration: line-through;"] [text x]) match.omitted)
              else [div [attribute "style" "white-space: nowrap;"] [
                text ((String.fromInt (List.length match.omitted)) ++ " omitted") ]])

            , if model.querySettings.annotateDebug
            then div [] [
              --div [ textSize Bulma.Modifiers.Typography.Small ] [
              --  text match.algorithm ]

              div [ textSize Bulma.Modifiers.Typography.Small ] [
                div [] [ text ("d " ++ (String.fromInt match.debug.document)) ]
                , div [] [ text ("s " ++ (String.fromInt match.debug.sentence)) ]
              ]
            ]
            else div [] []
          ]
        ]
      ]
      , mediaContent []
        [ span [ style "font-variant" "small-caps" ] [ text match.location.speaker ]
        , div [ pulledRight ]
          [
          small [] [ text (match.location.author ++ ", ") ]
          , small [ italicize ] [ text (match.location.work ++ ", ") ]
          , small [] [ text match.location.location ]
          ]
        , div [] [
          br [] []
          , br [] []

          , span []
          (List.intersperse (text " ") (List.map (\r -> regionView model r) (trimRegions match.regions)))
        ]
      ]
    ]

searchButton : Model -> Html Msg
searchButton model
  =
  let s = case model.search of
        NotSearching -> { state = Active, color = Bulma.Modifiers.Primary, text = "Search", msg = StartSearch }
        Searching _ -> { state = Active, color = Bulma.Modifiers.Danger, text = "Abort", msg = AbortSearch }
        SearchRequested -> { state = Loading, color = Bulma.Modifiers.Primary, text = "Search", msg = StartSearch }
        AbortRequested -> { state = Loading, color = Bulma.Modifiers.Danger, text = "Abort", msg = AbortSearch }
  in
    controlButton
    { buttonModifiers | state = s.state, color = s.color }
    []
    [ onClick s.msg ]
    [ text s.text ]

targetFloat : Json.Decoder (Maybe Float)
targetFloat =
  Json.map String.toFloat (Json.at ["target", "value"] Json.string)

onSliderInput : (Maybe Float -> msg) -> Attribute msg
onSliderInput tagger =
  on "input" (Json.map tagger targetFloat)

checked flag =
  if flag then
    [ Html.Attributes.attribute "checked" "checked" ]
  else
    []

optimizeAlignmentUI : QuerySettings -> Html Msg
optimizeAlignmentUI settings =
  card []
  [
    cardContent []
    [
      radio
      (
        [ onCheck (\_ -> (UpdateQuerySettings (CostCombine CombineMin))) ]
        ++ (checked (settings.costCombine == CombineMin))
      )
      "id-alignment-min"
      "weakest"
      , radio
      (
        [ onCheck (\_ -> (UpdateQuerySettings (CostCombine CombineSum))) ]
        ++ (checked (settings.costCombine == CombineSum))
      )
      "id-alignment-sum"
      "overall"
    ]
  ]

similarityMeasureRadioUI : QuerySettings -> Features -> (String,  String) -> Html Msg
similarityMeasureRadioUI settings features names =
  let
    (name, displayName) = names
    currentMeasure = settings.similarityMeasure
  in fields Left [] ([
    radio
    (
      [ onCheck (\_ -> (UpdateQuerySettings (UpdateSimilarityMeasure (Just { currentMeasure | name = name })))) ]
      ++ (checked (settings.similarityMeasure.name == name))
    )
    ( "id-similarity-measure-" ++ name)
    displayName
    ] ++ if features.quantiles then [
      optionalCheckbox (settings.similarityMeasure.name == name)
      (
        [
          onCheck (\x -> (UpdateQuerySettings (UpdateSimilarityMeasure (Just { currentMeasure | quantiles = x }))))
        ] ++ checked (settings.similarityMeasure.quantiles)
      )
      ("id-similarity-measure-quantiles-" ++ name)
      "quantiles"
    ]
    else [])


similarityMeasureUI : QuerySettings -> Features -> Html Msg
similarityMeasureUI settings features =
  let options = [("cosine", "Cosine")] ++ (if
        features.nicdm then [("nicdm", "NICDM")] else []) ++ (if
        features.apsynp then [("apsynp", "APSynP")] else []) ++ (if
        features.maximum then [("maximum", "Maximum")] else [])
  in
  card []
  [
    cardHeader []
    [
      cardTitle []
      [
        text "Embedding Similarity Measure"
      ]
    ]
    , cardContent []
    [
      div [] (List.map (\name -> (similarityMeasureRadioUI settings features name)) options)
    ]
  ]


similarityDetailsUI : QuerySettings -> Features -> Html Msg
similarityDetailsUI settings features =
  card []
  [
    cardHeader []
    [
      cardTitle []
      [
        text "Similarity Postprocessing"
      ]
    ]
    , cardContent []
    [
      span [ pulledLeft ]
      [ text "Falloff" ]
      , span [ pulledRight ]
      [ text ((Round.round 2 settings.similarityFalloff)) ]
      , falloffSlider
      [
        attribute "value" (String.fromFloat settings.similarityFalloff),
        onSliderInput (\x -> (UpdateQuerySettings (SimilarityFalloff x)))
      ]
      "slider-similarity-falloff",

      span [ pulledLeft ]
      [ text "Threshold" ]
      , span [ pulledRight ]
      [ text ((Round.round 0 settings.similarityThreshold) ++ "%") ]
      , fineGrainedSlider
      [
        attribute "value" (String.fromFloat settings.similarityThreshold),
        onSliderInput (\x -> (UpdateQuerySettings (SimilarityThreshold x)))
      ]
      "slider-similarity-threshold"
    ]
  ]

searchUI : Model -> Html Msg
searchUI model
  = container []
    [
      hero { heroModifiers | size = Bulma.Modifiers.Small, color = Bulma.Modifiers.White } []
      [ heroBody []
        [
           title H1 [] [ img [ src "static/vectorian.png" ] [] ]
        ]
      ]
      , container []
      [
        section NotSpaced []
        [
          fields Left []
          [
            controlInput { controlInputModifiers | expanded = True, disabled = (model.search /= NotSearching) }
              [ onKeyDown QueryKeyDown, onInput QueryInput ] [placeholder "Ask me and thou shall know."] []
            , searchButton model
          ]

          , div
            []
            (
              case model.search of
                Searching (Just progress) ->
                  [ easyProgress
                    { progressModifiers | size = Bulma.Modifiers.Small, color = Bulma.Modifiers.Primary }
                    []
                    progress
                  ]
                _ -> []
            )

          , section NotSpaced []
          [
            tileAncestor Auto []
            [
              tileChild Width4 []
              [
                card []
                [
                  cardHeader []
                  [
                    cardTitle []
                    [
                      text "Embedding Interpolation"
                    ]
                  ],
                  cardContent ( if model.querySettings.enableElmo then [invisible] else [] )
                  [
                    span [ pulledLeft ]
                    [ text "fasttext" ]
                    , span [ pulledRight ]
                    [ text "wnet2vec" ]
                    , slider
                    [
                      attribute "value" (String.fromFloat model.querySettings.mixEmbedding),
                      onSliderInput (\x -> (UpdateQuerySettings (MixEmbedding x)))
                    ]
                    "slider-mix-embedding"
                  ]
                ]
                , similarityMeasureUI model.querySettings model.features
                {- , card []
                [
                  cardContent []
                  [
                    checkbox
                    (
                      [
                        onCheck (\x -> (UpdateQuerySettings (EnableElmo x)))
                      ] ++ checked model.querySettings.enableElmo
                    )
                    "id-enable-elmo"
                    "Elmo"
                  ]
                ] -}
              ], tileChild Width4 []
              [
                -- optimizeAlignmentUI model.querySettings
                card []
                [
                  cardHeader []
                  [
                    cardTitle []
                    [
                      text "Alignment",

                      div [invisible] [
                        checkbox
                        (
                          [
                            onCheck (\x -> (UpdateQuerySettings (Bidirectional x)))
                          ] ++ checked model.querySettings.bidirectional
                        )
                        "id-bidirectional"
                        "Bidirectional"
                      ]
                    ]
                  ]
                  , cardContent []
                  [
                    span [ pulledLeft ]
                    [
                      text "Mismatch Length Penalty"
                    ]
                    , span [ pulledRight ]
                    [ text ("50% rel. after " ++ (String.fromInt model.querySettings.mismatchLengthPenalty) ++ " t.") ]
                    , mismatchCutoffSlider
                    [
                      attribute "value" (String.fromInt model.querySettings.mismatchLengthPenalty),
                      onSliderInput (\x ->
                       case x of
                           Just c -> (UpdateQuerySettings (MismatchLengthPenalty (Just (floor c))))
                           Nothing -> NoOp
                      )
                    ]
                    "slider-mismatch-length-cutoff",

                    mismatchPenaltyCurve model.querySettings.mismatchLengthPenalty 15,

                    div [ pulledLeft ]
                    [ text "Submatch Boosting" ]
                    , span [ pulledRight ]
                    [ text ((Round.round 0 model.querySettings.submatchWeight)) ]
                    , intSlider 10
                    [
                      attribute "value" (String.fromFloat model.querySettings.submatchWeight),
                      onSliderInput (\x -> (UpdateQuerySettings (SubmatchWeight x)))
                    ]
                    "slider-submatch-weight"

                  ]
                ]
              ], tileChild Width4 []
              [
                card []
                [
                  cardHeader []
                  [
                    cardTitle []
                    [
                      text "Part of Speech",

                      span [pulledRight] [
                        checkbox
                        (
                          [
                            onCheck (\x -> (UpdateQuerySettings (IgnoreDeterminers x)))
                          ] ++ checked model.querySettings.ignoreDeterminers
                        )
                        "id-ignore-determiners"
                        "Exclude Determiners"
                      ]

                    ]
                  ]
                  , cardContent []
                  [
                    div ( if model.querySettings.enableElmo then [invisible] else [] ) [
                      span [ pulledLeft ]
                      [
                        text "POS Mismatch Penalty"
                      ]
                      , span [ pulledRight ]
                      [ text ((Round.round 0 model.querySettings.posMismatch) ++ "%") ]
                      , slider
                      [
                        attribute "value" (String.fromFloat model.querySettings.posMismatch),
                        onSliderInput (\x -> (UpdateQuerySettings (PosMismatch x)))
                      ]
                      "slider-pos-mismatch" ],
                    div [] [
                      span [ pulledLeft ]
                      [
                        text "Semantic POS Weighting"
                      ]
                      , span [ pulledRight ]
                      [
                        text ((Round.round 0 model.querySettings.posWeighting) ++ "%")
                      ]
                      , slider
                      [
                        attribute "value" (String.fromFloat model.querySettings.posWeighting)
                        , onSliderInput (\x -> (UpdateQuerySettings (PosWeighting x)))
                      ]
                      "slider-pos-weighting"
                    ]
                  ]
                ],

                similarityDetailsUI model.querySettings model.features,

                if model.features.idf then card []
                [
                  cardHeader []
                  [
                    cardTitle []
                    [
                      text "Frequency"
                    ]
                  ], cardContent []
                  [
                    div [ pulledLeft ]
                    [ text "Inverse Frequency Scaling" ]
                    , span [ pulledRight ]
                    [ text ((Round.round 0 model.querySettings.idfWeight)) ]
                    , slider
                    [
                      attribute "value" (String.fromFloat model.querySettings.idfWeight),
                      onSliderInput (\x -> (UpdateQuerySettings (IDFWeight x)))
                    ]
                    "slider-idf-weight"
                  ]
                ] else div [] []
              ]
            ]
          ]

          , if List.isEmpty model.results
            then span [] []
            else section NotSpaced [] [
              fields Right []
              [
                checkbox
                (
                  [
                    onCheck (\x -> (UpdateQuerySettings (AnnotatePOS x)))
                  ] ++ checked model.querySettings.annotatePOS
                )
                "id-annotate-pos"
                "Annotations"

                {- , checkbox
                (
                  [
                    onCheck (\x -> (UpdateQuerySettings (AnnotateDebug x)))
                  ] ++ checked model.querySettings.annotateDebug
                )
                "id-annotate-debug"
                "Debug" -}
              ]
            ]

          , section NotSpaced []
          (List.map (\r -> matchView model r) (List.take 100 model.results))
        ]
      ]
    ]
