
"""

INFO 4300: Language and Information
Final Project: Color Palette Generator

Information Retrieval (IR) Scoring + Ranking Helper Functions

"""

import math
import urllib.request
import json
import ssl
from app.irsystem.controllers.rgb2lab import *
from app.irsystem.controllers.IR_main import *
from app.irsystem.controllers.search_controller import *


def deltaE(lab1, lab2):
    """
    Returns the perceptual distance between two colors in CIELAB.

    Params: lab1    color in LAB code [Tuple of Ints]
            lab2    color in LAB code [Tuple of Ints]
    """

    dL = lab1[0] - lab2[0]
    dA = lab1[1] - lab2[1]
    dB = lab1[2] - lab2[2]

    c1 = math.sqrt(lab1[1]**2 + lab1[2]**2)
    c2 = math.sqrt(lab2[1]**2 + lab2[2]**2)

    dC = c1 - c2
    dH = dA**2 + dB**2 - dC**2

    if dH < 0:
        dH = 0
    else:
        dH = math.sqrt(dH)

    sc = 1.0 + 0.045 * c1
    sh = 1.0 + 0.015 * c1

    dLKlsl = float(dL)
    dCkcsc = dC/sc
    dHkhsh = dH/sh

    dE = dLKlsl**2 + dCkcsc**2 + dHkhsh**2

    if dE < 0:
        return 0
    else:
        return math.sqrt(dE)


def colorDiff(c1, c2, code):
    """
    Returns the distance between two colors in either RGB space or HSL space.

    Params: c1      color in RGB, HSL, or LAB code [Tuple of Ints]
            c2      color in RGB, HSL, or LAB code [Tuple of Ints]
            code    'rgb', 'hsv' [String]
    """
    print("color diff")
    print(c1)
    print(c2)
    if code == 'rgb':
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)

    elif code == 'hsv':
        diff = (math.sin(c1[0])*c1[1]*c1[2] - math.sin(c2[0])*c2[1]*c2[2])**2
        diff += (math.cos(c1[0])*c1[1]*c1[2] - math.cos(c2[0])*c2[1]*c2[2])**2
        diff += (c1[2] - c2[2])**2
        return diff*100

    elif code == 'lab':
        return deltaE(c1, c2)


def getRGBDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the minimum RGB euclidian distances from the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    rgbDists = {}

    reqRGB = convertColor(reqColor, 'hex', 'rgb')

    for id, palette in palettes.items():
        minDist = 500
        for c in palette:
            cRGB = convertColor(c, 'hex', 'rgb')
            dist = colorDiff(reqRGB, cRGB, 'rgb')
            if dist < minDist:
                minDist = dist
        rgbDists[id] = minDist

    return rgbDists


def getHSVDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the minimum HSV cartesian distances from the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    hsvDists = {}

    reqHSV = convertColor(reqColor, 'hex', 'hsl')
    reqHSV = convertColor(reqHSV, 'hsl', 'hsv')

    for id, palette in palettes.items():
        minDist = 500
        for c in palette:
            cHSV = convertColor(c, 'hex', 'hsl')
            cHSV = convertColor(cHSV, 'hsl', 'hsv')
            dist = colorDiff(reqHSV, cHSV, 'hsv')
            if dist < minDist:
                minDist = dist
        hsvDists[id] = minDist

    return hsvDists


def getPerceptualDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the minimum perceptual distance of the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    deltaEDists = {}

    reqLAB = convertColor(reqColor, 'hex', 'rgb')
    reqLAB = convertColor(reqLAB, 'rgb', 'lab')

    for id, palette in palettes.items():
        minDist = 500
        for c in palette:
            cLAB = convertColor(c, 'hex', 'rgb')
            cLAB = convertColor(cLAB, 'rgb', 'lab')
            dist = colorDiff(reqLAB, cLAB, 'lab')
            if dist < minDist:
                minDist = dist
        deltaEDists[id] = minDist

    return deltaEDists


def CloseColorHelper(cymColors, colorToMatch):
    """
      Gets the closest color to the Cymbolism list of colors 
      based on the RGB distance

      Params: cymColors: list of 19 colors from the Cymbolism website (hexcodes)  = Clean - without hashtag 
              colorToMatch: one color from the palette to match 

      Returns: one of the 19 colors ( hexcode)

      """

    returnlist = {}
    colorToMatch = convertColor(colorToMatch, 'hex', 'rgb')
    for x in range(len(cymColors)):
        rgbcolor = convertColor(cymColors[x], 'hex', 'rgb')
        distance = colorDiff(rgbcolor, colorToMatch, 'rgb')
        returnlist[cymColors[x]] = distance

    larg = 0
    ret = ""
    for k, v in returnlist.items():
        if v > larg:
            ret = k
    return convertColor(ret, 'rgb', 'hex')


def keyword(userWords, paletteDict):
    """
      Returns a dictionary that includes the percentage score based on the colors and keywords 

      Params: userWords: the keywords that the user inputted matched 
      to a cymbolism words 
              paletteDict: dictionary of the palettes 
              data is dictionary where the key is the keyword, the value is list where each c

      Returns: Dictionary in format: {palette_id: average,...}
      """

    colordict = {}
    cymColors = list(cymColorsInvInd.keys())
    for palette in paletteDict.keys():
        score = 0
        for word in userWords:
            lst = []
            for color in paletteDict[palette]:
                closecolor = CloseColorHelper(cymColors, color)
                lst = cymData[word]
                ind = cymColorsInvInd[closecolor]
                colorScore = lst[ind]
            score += colorScore
        colordict[palette] = score
    return colordict


def convertColor(color, fromCode, toCode):
    """
    Returns a color converted from one code system to another. None if any
        params are incorrectly formatted.

    Ex: convertColor('(255,255,255)', 'rgb', 'hex') -> 'FFFFFF'

    Params: color       clean hexcode, rgb, hsl [String or Tuple of Ints]
            fromCode    'hex', 'rgb', 'hsl' [String]
            toCode      'hex', 'rgb', 'hsl', 'hsv', 'lab' [String]
    """
    if type(color) == str and "#" in color:
        color = clean_hex(color)

    if fromCode == 'hsl' and toCode == 'hsv':
        v = color[2]/100 + color[1]/100*min(color[2]/100, 1-color[2]/100)
        if v == 0:
            s = 0
        else:
            s = 2*(1 - color[2]/100/v)
        return (color[0], s, v)

    elif fromCode == 'rgb' and toCode == 'lab':
        return tuple(rgb2lab(color))

    query = '/id?' + fromCode + '=' + color
    url = 'https://www.thecolorapi.com' + query + '&format=json'
    print("URL")
    print(url)

    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)
    print("chekcing here")
    print(response)
    color_json = json.loads(response.read().decode())

    if None in color_json['rgb'].values():
        query = '/id?' + fromCode + '=' + color_json['hex']['clean']
        url = 'https://www.thecolorapi.com' + query + '&format=json'

        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(url, context=context)
        color_json = json.loads(response.read().decode())

    if toCode == 'hex':
        return color_json['hex']['clean']
    elif toCode == 'rgb':
        return (int(color_json['rgb']['r']), int(color_json['rgb']['g']),
                int(color_json['rgb']['b']))
    elif toCode == 'hsl':
        return (int(color_json['hsl']['h']), int(color_json['hsl']['s']),
                int(color_json['hsl']['l']))
    else:
        return None


#
