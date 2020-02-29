module BulmaExtensions exposing (..)

import Html exposing ( input, text, span )
import Html.Attributes exposing ( type_, class, attribute )

import Bulma.Form exposing (..)
import Bulma.Modifiers exposing ( .. )

switch attr id title =
    field []
    [
      input ([ attribute "id" id, type_ "checkbox", class "switch" ] ++ attr) []
      , Html.label [ attribute "for" id, unselectable ]
      [
        text title
      ]
    ]

-- note: setting the "name" here is important, otherwise radios won't be exclusive.

radio attr id title =
    field[]
    [
      input ([ attribute "id" id, type_ "radio", attribute "name" "bla", class "is-checkradio" ] ++ attr) []
      , Html.label [ attribute "for" id, unselectable ]
      [
        text title
      ]
    ]

checkbox attr id title =
    field []
    [
      input ([ attribute "id" id, type_ "checkbox", class "is-checkradio" ] ++ attr) []
      , Html.label [ attribute "for" id, unselectable ]
      [
        text title
      ]
    ]

optionalCheckbox visible attr id title =
    field (if visible then [] else [invisible])
    [
      input ([ attribute "id" id, type_ "checkbox", class "is-checkradio" ] ++ attr) []
      , Html.label [ attribute "for" id, unselectable ]
      [
        text title
      ]
    ]

slider attr id =
  input
  ([
    attribute "id" id
    , class "slider is-fullwidth"
    , attribute "min" "0"
    , attribute "max" "100"
    , attribute "step" "25"
    , type_ "range"
  ] ++ attr) []

fineGrainedSlider attr id =
  input
  ([
    attribute "id" id
    , class "slider is-fullwidth"
    , attribute "min" "0"
    , attribute "max" "100"
    , attribute "step" "1"
    , type_ "range"
  ] ++ attr) []

intSlider max attr id =
  input
  ([
    attribute "id" id
    , class "slider is-fullwidth"
    , attribute "min" "0"
    , attribute "max" (String.fromInt max)
    , attribute "step" "1"
    , type_ "range"
  ] ++ attr) []


mismatchCutoffSlider attr id =
  input
  ([
    attribute "id" id
    , class "slider is-fullwidth"
    , attribute "min" "1"
    , attribute "max" "30"
    , attribute "step" "1"
    , type_ "range"
  ] ++ attr) []

falloffSlider attr id =
  input
  ([
    attribute "id" id
    , class "slider is-fullwidth"
    , attribute "min" "0.25"
    , attribute "max" "5"
    , attribute "step" "0.25"
    , type_ "range"
  ] ++ attr) []
