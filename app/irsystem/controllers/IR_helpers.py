
"""

INFO 4300: Language and Information
Final Project: Color Palette Generator

Information Retrieval (IR) Scoring + Ranking Helper Functions

"""

import math
import urllib.request, json
import rgb2lab


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

    dLKlsl = float(dL);
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

    for id,palette in palettes.items():
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

    for id,palette in palettes.items():
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

    for id,palette in palettes.items():
        minDist = 500
        for c in palette:
            cLAB = convertColor(c, 'hex', 'rgb')
            cLAB = convertColor(cLAB, 'rgb', 'lab')
            dist = colorDiff(reqLAB, cLAB, 'lab')
            if dist < minDist:
                minDist = dist
        deltaEDists[id] = minDist

    return deltaEDists


def getAvgPercepDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the average perceptual distance of the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    deltaEDists = {}

    reqLAB = convertColor(reqColor, 'hex', 'rgb')
    reqLAB = convertColor(reqLAB, 'rgb', 'lab')

    for id,palette in palettes.items():
        sumDist = 0
        for c in palette:
            cLAB = convertColor(c, 'hex', 'rgb')
            cLAB = convertColor(cLAB, 'rgb', 'lab')
            sumDist += colorDiff(reqLAB, cLAB, 'lab')
        deltaEDists[id] = sumDist/len(palette)

    return deltaEDists


def convertColor(color, fromCode, toCode):
    """
    Returns a color converted from one code system to another. None if any
        params are incorrectly formatted.

    Ex: convertColor('(255,255,255)', 'rgb', 'hex') -> 'FFFFFF'

    Params: color       clean hexcode, rgb, hsl [String or Tuple of Ints]
            fromCode    'hex', 'rgb', 'hsl' [String]
            toCode      'hex', 'rgb', 'hsl', 'hsv', 'lab' [String]
    """

    if fromCode == 'hsl' and toCode == 'hsv':
        v = color[2]/100 + color[1]/100*min(color[2]/100, 1-color[2]/100)
        if v == 0:
            s = 0
        else:
            s = 2*(1 - color[2]/100/v)
        return (color[0], s, v)

    elif fromCode == 'rgb' and toCode == 'lab':
        return tuple(rgb2lab.rgb2lab(color))

    query = '/id?' + fromCode + '=' + color
    url = 'https://www.thecolorapi.com' + query + '&format=json'

    response = urllib.request.urlopen(url)
    color_json = json.loads(response.read().decode())

    if None in color_json['rgb'].values():
        query = '/id?' + fromCode + '=' + color_json['hex']['clean']
        url = 'https://www.thecolorapi.com' + query + '&format=json'

        response = urllib.request.urlopen(url)
        color_json = json.loads(response.read().decode())

    if toCode == 'hex':
        return color_json['hex']['clean']
    elif toCode == 'rgb':
        return (int(color_json['rgb']['r']), int(color_json['rgb']['g']),
            int(color_json['rgb']['b']))
    elif toCode == 'hsl':
        return (int(color_json['hsv']['h']), int(color_json['hsv']['s']),
            int(color_json['hsv']['v']))
    else:
        return None









#
