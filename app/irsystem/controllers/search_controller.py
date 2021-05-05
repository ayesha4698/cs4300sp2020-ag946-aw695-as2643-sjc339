from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import time
import math
import random
from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# from app.irsystem.controllers.IR_main import *
# from app.irsystem.controllers.IR_helpers import *
from app.irsystem.controllers.rgb2lab import *
from app.irsystem.controllers.cossim import *


import numpy as np
import pandas as pd
from colormap import rgb2hex
import re
import requests
import csv
from csv import writer
import json
import colorsys
import nltk
import itertools
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


###########################################################################################
#                                          GLOBALS                                        #
###########################################################################################

# dataset globals
cymData = {}
cymColorsInvInd = {}
cymWordsInvInd = {}
cymVotes = {}

# /Users/ayesha/cs4300sp2020-ag946-aw695-as2643-sjc339/app/irsystem/controllers/IR_main.py
with open('data/Cymbolism.csv', mode='r') as infile:
    reader = csv.reader(infile)
    ind = 1
    for rows in reader:
        cymData[rows[0]] = rows[1:-1]
        cymVotes[rows[0]] = rows[-1]
        cymWordsInvInd[rows[0]] = ind
        ind += 1

for i in range(len(cymData['word'])):
    color = cymData['word'][i]
    colorName = color[color.index(' ')+2:]
    cymColorsInvInd[colorName] = i

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

netid = "Alisa Wong (aw695), Anishka Singh (as2643), Ayesha Gagguturi (ag946), Joy Thean (jct263), and Skylar Capasso (sjc339)"


###########################################################################################
#                                       IR HELPERS                                        #
###########################################################################################

def to_wordnet(tag):
    """
    Returns a wordnet PartOfSpeech object. None if no valid wordnet tag.
    Params: tag     wordnet PartOfSpeech string
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None


def searchCymDfns(w1):
    """
    Returns the similarity score and Cymbolism word match for a given Synset object.
    This function searches all the definitions of all the Cymbolism words to
    find the highest similarity score with the given keyword.

    Params: w1      Synset object
    """
    match = ''
    maxSim = 0.0
    # dummy run for speed
    # arbitrary word that doesn't have match in dataset
    wt = wordnet.synsets('cool')[-1]
    for cymW in cymData.keys():
        if cymW != 'word':
            cymW = cymW.replace(" ", "_")
            for w2 in wordnet.synsets(cymW):
                sim = wt.wup_similarity(w2)
                if sim is None:
                    sim = wt.path_similarity(w2)
                if sim is not None and sim > maxSim:
                    maxSim = sim
                    match = cymW

    # real run
    for cymW in cymData.keys():
        if cymW != 'word':
            cymW = cymW.replace(" ", "_")
            for w2 in wordnet.synsets(cymW):
                sim = w1.wup_similarity(w2)
                if sim is None:
                    sim = w1.path_similarity(w2)
                if sim is not None and sim > maxSim:
                    maxSim = sim
                    match = cymW

    maxSim = max([0, maxSim - .1])                  # adjust? fix later
    return maxSim, match


def searchKeywordDfn(w1):
    """
    Returns the similarity score and Cymbolism word match for a given Synset object.
    This function searches all the definitions of all the Cymbolism words to
    find the highest similarity score with the given keyword.

    Params: w1      Synset object
    """
    match = ''
    maxSim = 0.0
    toks = tokenize(w1.definition())
    toks = [w for w in toks if not w in stop_words]

    tagged = nltk.pos_tag(toks)
    lemmatzr = WordNetLemmatizer()

    synsets = []
    n = []
    v = []
    a = []
    r = []
    s = []

    for tok in tagged:
        wn_tag = to_wordnet(tok[1])
        if not wn_tag:
            continue

        lemma = lemmatzr.lemmatize(tok[0], pos=wn_tag)
        if wordnet.synsets(lemma, pos=wn_tag) != []:
            word = wordnet.synsets(lemma, pos=wn_tag)[0]
        else:
            word = wordnet.synsets(lemma)[0]

        if word.pos() == 'n':
            n = n + [word]
        elif word.pos() == 'v':
            v = v + [word]
        elif word.pos() == 'a':
            a = a + [word]
        elif word.pos() == 'r':
            r = r + [word]
        elif word.pos() == 's':
            s = s + [word]
        else:
            synsets = synsets + [word]

    synsets = a + n + s + r + v + synsets

    for w3 in synsets:
        for cymW in cymData.keys():
            if cymW != 'word':
                cymW = cymW.replace(" ", "_")
                for w2 in wordnet.synsets(cymW):
                    sim = w3.wup_similarity(w2)
                    if sim is None:
                        sim = w3.path_similarity(w2)
                    if sim is not None and sim > maxSim:
                        maxSim = sim
                        match = cymW

    maxSim = max([0, maxSim - .2])                  # adjust? fix later
    return maxSim, match


def getSynset(dfn):
    """
    Returns the Synset object that matches the word and
        definition inputted by the user.

    Params: dfn     User's keyword in the format:
                        'word - definition' [String]
    """
    # print('DFN_-----------------')
    # print(dfn)
    # print(dfn.find("-"))
    if not dfn.find("-") == -1:
        kw = dfn[:dfn.index("-")-1]         # skip the space

        kw = stemmer.stem(kw)
        # print(kw)
        syns = wordnet.synsets(kw.replace(" ", "_"))
        #print(syns, kw)
        if syns == []:
            kw = dfn[:dfn.index("-")-1]
            syns = wordnet.synsets(kw.replace(" ", "_"))

        query = dfn[dfn.index("-")+2:]      # skip the space
    else:
        kw = stemmer.stem(dfn)
        syns = wordnet.synsets(kw.replace(" ", "_"))
        if syns == []:
            return wordnet.synsets(dfn.replace(" ", "_"))[0]
        else:
            return syns[0]

    # find Synset object based on definition
    msgs = []
    for syn in syns:
        toks = tokenize(syn.definition())
        msgs.append({'toks': toks})

    inv_idx = build_inverted_index(msgs)
    idf = compute_idf(inv_idx, len(msgs))
    inv_idx = {key: val for key, val in inv_idx.items()
               if key in idf}
    doc_norms = compute_doc_norms(inv_idx, idf, len(msgs))
    ind_search = index_search(query, inv_idx, idf, doc_norms)

    msg_id = ind_search[0][1]
    return syns[msg_id]


def keywordMatch(dfns):
    """
    Returns a list of Cymbolism keywords that match to each keyword of the user's
        input, of the following format:
        [(keyword, similarity score),
        ...
        (keyword, similarity score)]
    Params: dfns    List of user's keywords where each string is formatted:
                        'word - definition' [List of Strings]
    """
    synwords = []
    keywords = []
    wordMatch = {}

    for dfn in dfns:
        if dfn in cymData.keys() and dfn != 'word':
            s = wordnet.synsets(dfn)[0]
        else:
            s = getSynset(dfn)
        synwords.append(s)
        if not dfn.find("-") == -1:
            kw = dfn[:dfn.index("-")-1]
        else:
            kw = dfn
        wordMatch[kw] = s

    words = list(wordMatch.keys())
    wordsInd = 0
    for w1 in synwords:
        match = ''
        maxSim = 0.0
        tries = 10
        syn_tries = []
        got_first = False

        while tries > 0 and not got_first:
            # keyword or keyword's synonyms are in Cymbolism
            if maxSim == 0.0:
                for lem in w1.lemmas():
                    if lem.name() in cymData.keys() and lem.name() != 'word':
                        maxSim = 1.0
                        match = lem.name()
                        got_first = True
                        break

            # keyword matches Cymbolism word's meanings
            if maxSim == 0.0:
                maxSim, match = searchCymDfns(w1)

            # keyword's definition matches Cymbolism word's meanings
            if maxSim == 0.0:
                maxSim, match = searchKeywordDfn(w1)

            # still no match, use random definition
            if maxSim <= 60 and not got_first:
                nam = w1.name()
                syns = wordnet.synsets(nam[:nam.index('.')])
                w1 = syns[random.randint(0, len(syns)-1)]
                wordMatch[words[wordsInd]] = w1

            syn_tries.append((match, maxSim))
            tries -= 1

        if len(syn_tries) > 1:
            sorted_tries = sorted(syn_tries, key=lambda x: x[1], reverse=True)
            match = sorted_tries[0][0]
            maxSim = sorted_tries[0][1]

        if match != '':
            for word, syn in wordMatch.items():
                if type(syn) is not tuple and w1 == syn:
                    wordMatch[word] = (match, maxSim)
            keywords.append((match, maxSim))
        else:
            wordMatch.pop(words[wordsInd], None)

        wordsInd += 1

    print("\nKEYWORDS MATCH")
    print(wordMatch)

    return keywords, wordMatch


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

    # print("RGBDIST")
    # print(palettes)
    # print(reqColor)

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

    small = 1000
    ret = ""
    for k, v in returnlist.items():
        if v < small:
            small = v
            ret = k
    return ret


def keyword(userWords, paletteDict, wordMatch):
    """
      Returns one dictionary that includes the percentage score based on the colors and keywords
      Returns another dictionary that also includes the keyword and the corresponding percentage score
      for each palette
      Params: userWords: the keywords that the user inputted matched
      to a cymbolism words
              paletteDict: dictionary of the palettes
              data is dictionary where the key is the keyword, the value is list where each c
      Returns: Dictionary in format: {palette_id: average,...}
                                    {palette_id: [(keyword, percent), (keyword, percent),...],...}
      """

    colordict = {}
    keywordDict = {}
    wordsDict = {}
    cymColors = list(cymColorsInvInd.keys())
    maxScore = 0
    for palette in paletteDict.keys():
        score = 0
        keywordDict[palette] = []
        for word in userWords:
            wordScore = 0
            if word not in wordsDict:
                wordsDict[word] = 0

            lst = []
            for color in paletteDict[palette]:
                closecolor = CloseColorHelper(cymColors, color)
                lst = cymData[word]
                ind = cymColorsInvInd[closecolor]
                colorScore = float(lst[ind])
                wordScore += colorScore
            score += wordScore
            keywordDict[palette].append((word, wordScore))
            if wordScore > wordsDict[word]:
                wordsDict[word] = wordScore

        if score > maxScore:
            maxScore = score
        colordict[palette] = score

    for id in colordict:
        colordict[id] /= maxScore

    for id, lst in keywordDict.items():
        for i in range(len(lst)):
            word = keywordDict[id][i][0]
            percent = keywordDict[id][i][1]
            percent += (100 - percent)*(wordsDict[word]/100)/2
            if percent >= 100:
                percent = keywordDict[id][i][1]
            keywordDict[id][i] = (word, percent)

    keywordBreakdown = {}
    for id, lst in keywordDict.items():
        keywordBreakdown[id] = []
        for tup in lst:
            for orig, cym in wordMatch.items():
                if cym[0] == tup[0]:
                    score = tup[1]*cym[1]
                    new_tup = (orig, tup[0], score)
                    keywordBreakdown[id].append(new_tup)

    return colordict, keywordBreakdown


def convertColor(color, fromCode, toCode):
    """
    Returns a color converted from one code system to another. None if any
        params are incorrectly formatted. (all clean hexcodes)
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

    elif fromCode == 'rgb' and toCode == 'hsv':
        return colorsys.rgb_to_hsv(color[0], color[1], color[2])

    elif fromCode == 'rgb' and toCode == 'hsl':
        newcolor = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        return (newcolor[0], newcolor[2], newcolor[1])

    elif fromCode == 'rgb' and toCode == 'hex':
        return '%02x%02x%02x' % color

    elif fromCode == 'hex' and toCode == 'rgb':
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    elif fromCode == 'hex' and toCode == 'hsl':
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        newcolor = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        return (newcolor[0], newcolor[2], newcolor[1])

    else:
        raise ValueError('Invalid inputs to convertColor: ' + str(color) + ', '
                         + fromCode + ', ' + toCode)


def rgbToHsl(color):
    r = color[0] / 255
    g = color[1] / 255
    b = color[2] / 255

    cMax = max(r, g, b)
    cMin = min(r, g, b)

    delta = cMax - cMin
    l = (cMax + cMin) / 2
    h = 0
    s = 0

    if delta == 0:
        h = 0
    elif cMax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cMax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    if delta == 0:
        s = 0
    else:
        s = delta / (1 - abs(2 * l - 1))

    hsl = (h, s, l)
    return hsl


def normalize_rbg(color, m):

    color = math.floor((color + m) * 255)

    if color < 0:
        color = 0
    elif color > 255:
        color = 255

    return color


def hslToRgb(color):

    h = color[0]
    s = color[1]
    l = color[2]

    c = (1 - abs(2 * l - 1) * s)
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    r, g, b = 0, 0, 0

    if h < 60:
        r = c
        g = x
        b = 0
    elif h < 120:
        r = x
        g = c
        b = 0
    elif h < 180:
        r = 0
        g = c
        b = x
    elif h < 240:
        r = 0
        g = x
        b = c
    elif h < 300:
        r = x
        g = 0
        b = c
    else:
        r = c
        g = 0
        b = x

    r = normalize_rbg(r, m)
    g = normalize_rbg(g, m)
    b = normalize_rbg(b, m)

    return (r, g, b)


def hue_adjuster(color, degree):
    # print()
    # rgb_from_hex = convertColor(color, 'hex', 'rgb')

    # r1 = rgb_from_hex[0]
    # g1 = rgb_from_hex[1]
    # b1 = rgb_from_hex[2]

    # hsl = rgbToHsl(rgb_from_hex)
    # rgb = hslToRgb(hsl)

    # h = hsl[0]
    # s = hsl[1]
    # l = hsl[2]

    # h += degree
    # if h > 360:
    #     h -= 360
    # elif h < 0:
    #     h += 360

    # new_hsl = (h, s, l)

    # new_rgb = hslToRgb(new_hsl)
    # r2 = new_rgb[0]
    # g2 = new_rgb[1]
    # b2 = new_rgb[2]

    # if r1 == g1 and r1 == b1:
    #     avg = (r2 + g2 + b2) // 3
    #     r2 = avg
    #     g2 = avg
    #     b2 = avg
    # new_rgb = (r2, g2, b2)
    # print(new_rgb)
    # new_hex = convertColor(new_rgb, 'rgb', 'hex').upper()

    # return new_hex

    rgb_from_hex = convertColor(color, 'hex', 'rgb')
    print(rgb_from_hex)
    if rgb_from_hex[0] == rgb_from_hex[1] == rgb_from_hex[2]:
        val = (rgb_from_hex[0] + degree) % 255
        print(val)
        new_rgb = (val, val, val)

    else:

        hsl = rgbToHsl(rgb_from_hex)
        rgb = hslToRgb(hsl)

        h = hsl[0]
        s = hsl[1]
        l = hsl[2]

        h += degree
        if h > 360:
            h -= 360
        elif h < 0:
            h += 360

        new_hsl = (h, s, l)

        new_rgb = hslToRgb(new_hsl)

    new_hex = convertColor(new_rgb, 'rgb', 'hex').upper()
    return new_hex


def isClose(palette1, palette2, threshold):
    """
    Check if the palette is close enough to palette that is already saved in the csv
    If it is already in the csv, return true otherwise return false

    Params:
            palette1 -  a palette from the CSV
            palette2 - a palette to compare

    Returns: True if the palette is already in the csv or is below the threshold
    and is similar to a palette already in the csv
            False if the palette does not exist or the distance is above the threshold
    """
    endDict = {}
    lst = []
    if palette1 == palette2:
        return True
    for c1 in palette2:
        minScore = []
        for c2 in palette1:
            ca = convertColor(c1, 'hex', 'rgb')
            cb = convertColor(c2, 'hex', 'rgb')
            score = colorDiff(ca, cb, "rgb")
            # print(score)
            minScore.append(score)
        lst.append(minScore)
    # print("minScore", minScore)
    avg = sum(minScore)/len(minScore)
    print("avg", avg)
    # threshold = 115
    if avg < threshold:
        return True
    else:
        return False
    # print("same?", palette1, palette2)
    # return palette1 == palette1


def closestPalette(palette):
  # loop throught the csv palettes 
    with open('data/votes.csv', 'r', newline='') as file:
        myreader = csv.reader(file, delimiter=',')
        for rows in myreader:
            if rows[0] != 'Palette':
                paletteToCompare = []
                [paletteToCompare.extend(rows[0].split( ))] 
                if(isClose(paletteToCompare, palette, 140)):
                    print("close" , paletteToCompare)
                    return paletteToCompare, True
        
    print("not close")
    return palette, False

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
cymbolism = ['abuse', 'accessible', 'addiction', 'agile', 'amusing', 'anger', 'anticipation', 'art deco', 'authentic', 'authority', 'average', 'baby', 'beach', 'beauty', 'beer', 'benign', 'bitter', 'blend', 'blissful', 'bold', 'book', 'boss', 'brooklyn', 'busy', 'calming', 'capable', 'car', 'cat', 'certain', 'charity', 'cheerful', 'chicago', 'classic', 'classy', 'clean', 'cold', 'colonial', 'comfort', 'commerce', 'compelling', 'competent', 'confident', 'consequence', 'conservative', 'contemporary', 'cookie', 'corporate', 'cottage', 'crass', 'creative', 'cute', 'dance', 'dangerous', 'decadent', 'decisive', 'deep', 'devil', 'discount', 'disgust', 'dismal', 'dog', 'drunk', 'dublin', 'duty', 'dynamic', 'earthy', 'easy', 'eclectic', 'efficient', 'elegant', 'elite', 'enduring', 'energetic', 'entrepreneur', 'environmental', 'erotic', 'excited', 'expensive', 'experience', 'fall', 'familiar', 'fast', 'fear', 'female', 'football', 'freedom', 'fresh', 'friendly', 'fun', 'furniture', 'future', 'gay', 'generic', 'georgian', 'gloomy', 'god', 'good', 'goth', 'government', 'grace', 'great', 'grow', 'happy', 'hard', 'hate', 'hazardous', 'hippie', 'hockey', 'honor', 'hope', 'hot', 'hunting', 'hurt', 'hygienic', 'ignorant', 'imagination', 'impossible', 'improbable', 'influence', 'influential', 'insecure', 'inviting', 'invulnerable', 'jacobean', 'jealous', 'joy', 'jubilant', 'junkie', 'knowledge', 'kudos', 'launch', 'lazy', 'leader', 'liberal', 'library', 'light', 'likely', 'lonely', 'love', 'magic', 'marriage', 'maximal', 'mean', 'medicine', 'melancholy', 'mellow', 'minimal', 'mission', 'modern', 'moment', 'money', 'music', 'mystical', 'narcissist', 'natural', 'naughty', 'new', 'nimble', 'now', 'objective', 'old', 'optimistic', 'organic', 'paradise', 'party', 'passion', 'passive', 'peace', 'peaceful', 'personal', 'playful', 'pleasing', 'possible', 'powerful', 'preceding', 'predatory', 'prime', 'probable', 'productive', 'professional', 'profit', 'progress', 'public', 'pure', 'radical', 'railway', 'rain', 'real', 'rebellious', 'recession', 'reconciliation', 'recovery', 'relaxed', 'reliability', 'retro', 'rich', 'risk', 'rococo', 'romantic', 'royal', 'rustic', 'sad', 'sadness', 'safe', 'sarcasm', 'secure', 'sensible', 'sensual', 'sex', 'shabby', 'silly', 'simple', 'slow', 'smart', 'smooth', 'snorkel', 'soft', 'solar', 'sold', 'solid', 'somber', 'spiffy', 'sport', 'spring', 'stability', 'star', 'strong', 'studio', 'style', 'stylish', 'submit', 'suburban', 'success', 'summer', 'sun', 'sunny', 'surprise', 'sweet', 'symbol', 'tasty', 'therapeutic', 'threat', 'time', 'tomorrow', 'treason', 'trust', 'trustworthy', 'uncertain', 'uniform', 'unlikely', 'unsafe', 'urban', 'value', 'vanity', 'victorian', 'vitamin', 'vulnerability', 'vulnerable', 'war', 'warm', 'winter', 'wise', 'wish', 'work', 'worm', 'young'];
invertedIndex = {}
for i, word in enumerate(cymbolism):
  invertedIndex[word] = i 

def paletteToCSV(palette, keywords, vote):
    CSVpalette, found = closestPalette(palette)
    CSVpalette = (' '.join([str(elem) for elem in CSVpalette])).replace(",", " ")
    if not found : 
        votes =  "1 "+ str(vote)
        row_contents = [CSVpalette]
        keywordsVotes = np.empty(len(cymbolism), dtype=object)
        for i in range(len(keywordsVotes)):
            keywordsVotes[i] = '0 0'
        for word in keywords:
            keywordsVotes[invertedIndex[word]] = votes
        row_contents += list(keywordsVotes)
        # Append a list as new line to an old csv file
        append_list_as_row('data/votes.csv', row_contents)

    else: 
        # already in csv
        keywordsVotes = []
        with open('data/votes.csv', 'r', newline='') as file:
            myreader = csv.reader(file, delimiter=',')
            for rows in myreader:
                if rows[0] == CSVpalette:
                    for idword in range(len(cymbolism)):
                        if(cymbolism[idword] in keywords):
                            currVotesArray = rows[idword+1].split()
                            totalVotes = int(currVotesArray[0]) + 1
                            netVotes = int(currVotesArray[1]) + int(vote)
                            currVotesString = str(totalVotes) + " " + str(netVotes)
                            keywordsVotes.append(currVotesString)
                        else:
                            keywordsVotes.append(rows[idword+1])
                    break
           
        # reading the csv file
        df = pd.read_csv("data/votes.csv")
        for i, word in enumerate(cymbolism):
            df.loc[df["Palette"]== CSVpalette, word] = keywordsVotes[i]

        df.to_csv("data/votes.csv", index=False)
    return None

def energy_adjust(color, energy):
    """ adjusts a hex code so that it’s brighter or more muted
    :param hex: string
    :param brightness: decimal for the percentage of how bright
    Input - 5 * 2 * 10 (test how large increase should be)
    Return: hex code
"""

    energy -= 5

    rgb = convertColor(color, 'hex', 'rgb')

    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    if energy > 0:
        energy *= 10
        newR = r + (256 - r) * energy // 100
        newG = g + (256 - g) * energy // 100
        newB = b + (256 - b) * energy // 100

        newRGB = (abs(newR), abs(newG), abs(newB))
    elif energy < 0:
        energy /= 10
        newR = round(min(max(0, r + (r * energy)), 255))
        newG = round(min(max(0, g + (g * energy)), 255))
        newB = round(min(max(0, b + (b * energy)), 255))

        newRGB = (abs(newR), abs(newG), abs(newB))
    else:
        newRGB = rgb

    newHex = convertColor(newRGB, 'rgb', 'hex')
    # print('NEWHEX')
    # print(newHex)
    newHex = newHex.upper()
    print(newHex)
    return newHex


def clean_hex(color):
    """
      This function converts the color names in the dataset to just be a hex number
      For example:
        'olive #808000' -> 808000
      Params: color     string of form 'color #hex'
    """
    split = color.find('#')
    return color[split+1:]


def parse_data():
    """
      This function parses the dataset. It returns a dictionary in which the
      keys are the words from Cymbolism. The values of each key is a list
      of tuples (unsorted) where the second element of the tuple is a color
      and the first element is the score of that color.
    """
    with open("data/Cymbolism.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        word_dict = {}
        count = 0
        color_index = {}

        total_votes = {}
        for row in spamreader:

            votes_index = len(row) - 1
            if count == 0:
                index = 0
                for color in row:
                    if color != 'word':
                        color_index[index + 1] = color
                        index += 1
            else:

                index = 0
                word = ''
                for value in row:
                    if index == 0:
                        word = row[0]
                    elif index == votes_index:
                        total_votes[word] = float(value)
                    else:
                        if word in word_dict:
                            word_dict[word].append(
                                (float(row[index]), clean_hex(color_index[index])))
                        else:
                            word_dict[word] = [
                                (float(row[index]), clean_hex(color_index[index]))]
                    index += 1

            count += 1

    return word_dict, total_votes


def normalize_score(score, num_votes, total_votes):
    """
        :param score: float
        :param necessary_color: list of hex codes
        :param energy:
        :return: normalized score (float)
    """
    # #print((score * num_votes) / total_votes)
    return (score * num_votes) / total_votes


def top_colors_from_keywords(keywords, energy):
    """ Generating each keywords top color from the dataset
        :param keywords: tuple (keyword (str), similarity score (float))
        :param energy: string of energy level
        :return: list tuples (top colors with energy adjusted (hex codes), score of color)
    """
    top_colors = {}

    word_dict, total_votes = parse_data()
    sorted_word_dict = {}

    for word, lst in word_dict.items():
        sort = sorted(lst, reverse=True)
        sorted_word_dict[word] = sort

    total_votes_num = 0
    for tup in keywords:
        word = tup[0]
        # print("----------WORDS")
        # print(keywords)
        # #print(total_votes)
        # print(word)
        total_votes_num += total_votes[word]

    for tup in keywords:
        word = tup[0]
        sim_score = tup[1]
        if word in sorted_word_dict:
            above_thresh = True
            while above_thresh:
                tups = sorted_word_dict[word]
                for tup in tups:
                    score = tup[0]
                    color = tup[1]
                    if score >= 10:
                        if energy_adjust(color, energy) in top_colors:
                            top_colors[energy_adjust(color, energy)] += sim_score * normalize_score(
                                (score), total_votes[word], total_votes_num)
                        else:
                            top_colors[energy_adjust(color, energy)] = sim_score * normalize_score(
                                (score), total_votes[word], total_votes_num)
                    else:
                        above_thresh = False

    top_colors_hues = {}
    for color, energy in top_colors.items():
        # print('HERE')
        # print(color)
        # if len(top_colors) < 2:
        # hue1 = hue_adjuster(color, 20)
        # hue2 = hue_adjuster(color, -20)
        # hue3 = hue_adjuster(color, 30)
        # hue4 = hue_adjuster(color, -30)
        # top_colors_hues[color] = [color, hue1, hue2, hue3, hue4]
        # else:
        if len(top_colors) > 2:
            print(color)
            hue1 = hue_adjuster(color, 10)
            hue2 = hue_adjuster(color, -10)
            energy1 = energy_adjust(color, -5)
            energy2 = energy_adjust(color, 5)
            # hue3 = hue_adjuster(color, 30)
            # hue4 = hue_adjuster(color, -30)
            top_colors_hues[color] = [color, hue1, hue2, energy1, energy2]
        else:
            hue1 = hue_adjuster(color, 20)
            hue2 = hue_adjuster(color, -20)
            hue3 = hue_adjuster(color, 30)
            hue4 = hue_adjuster(color, -30)
            energy1 = energy_adjust(color, -10)
            energy2 = energy_adjust(color, 10)
            # hue3 = hue_adjuster(color, 30)
            # hue4 = hue_adjuster(color, -30)
            top_colors_hues[color] = [color, hue1,
                                      hue2, hue3, hue4, energy1, energy2]
    # print('----- TOP COLORS -----')
    # print(top_colors_hues)
    # return top_colors
    print(top_colors)
    return top_colors, top_colors_hues


def palette_generator(hex_codes, n, energy):
    """ Adds on the “N” values to the list of input colors and outputs the api generated palette
    need to convert inputs to rgb and outputs to hex
:param hex_codes: list of hex colors (up to three)
    :param n: number of colors
:return: list of hex color codes computed from the Colormind API
"""

    #print("HEX CODES")
    # print(hex_codes)

    url = "http://colormind.io/api/"

    input_lst = []
    for hex in hex_codes:
        print(hex)
        # print('---------HEX--------')
        # print(hex)
        input_lst.append(convertColor(hex, "hex", "rgb"))
        print(hex)
        # input_lst.append(ImageColor.getcolor(hex, "RGB"))
    # print(input_lst)

    n = len(input_lst)
    for add_color in range(n, 5):
        input_lst.append("N")

    #print("INPUT LIST")
    # print(input_lst)
    data = {
        "model": "default",
        "input": input_lst
    }
    # print('DATA')
    # print(data)
    start = time.time()
    res = requests.post(url, data=json.dumps(data))
    print('API TIMES')
    print(time.time() - start)

    hex = []
    for rgb in res.json()['result']:
        color = rgb2hex(rgb[0], rgb[1], rgb[2])
        # print('---Color----')

        # print(color)
        # hex.append(energy_adjust(color, energy))

        hex.append(clean_hex(color))

    # print('HEX')
    # print(hex)
    return hex


def create_combo_hex_codes(top_keywords_color_lst, necessary_color_lst, energy):
    """ Calls the palette generator helper
:param keywords: list of list top keywords colors
:param necessary_color: list of list hex codes
    :param
:return: a list of lists of hex color codes
"""
    combo_hex_code_lst = []

    #print('necc', necessary_color_lst)

    necessary_colors = []

    for c in necessary_color_lst:
        necessary_colors.append((c, 100))
    # print('HERE')
    # print(top_keywords_color_lst)
    keyword_lst = list(set(top_keywords_color_lst.keys()))
    # print(keyword_lst)
    # start = time.time()
    combinations_object = list(itertools.combinations(keyword_lst, 2))

    # print('object')
    # print(combinations_object)
    #print('combinations time')
    #print(time.time() - start)

    # combo_short = combinations_object[:5]

    if(len(combinations_object) > 100):
        combinations_object = combinations_object[:100]

    # print(len(combinations_object))

    for i in range(len(combinations_object)):
        n = len(combinations_object)+len(necessary_color_lst)
        hex_codes = necessary_colors + list(combinations_object[i])

        # start = time.time()
        # print("HEX")
        # print(hex_codes)

        combo_hex_code_lst.append(palette_generator(hex_codes, n, energy))
        #print('palette generator time')
        #print(time.time() - start)
    # for i in range(len(combo_short)):
    #     n = len(combo_short)+len(necessary_color_lst)
    #     hex_codes = necessary_color_lst + list(combo_short[i])

    #     start = time.time()

    #     combo_hex_code_lst.append(palette_generator(hex_codes, n))
    #     #print('palette generator time')
    #     #print(time.time() - start)

    return combo_hex_code_lst


def create_combinations(top_colors, necessary_colors, top_colors_hues, energy):
    combinations = []
    seen = {}
    if len(top_colors) > 2:
        # print('LENGTH')
        # print(len(top_colors))
        for color1, lst1 in top_colors_hues.items():
            for color2, lst2 in top_colors_hues.items():
                if color1 != color2:
                    for hue1 in lst1:
                        for hue2 in lst2:
                            if hue1 != hue2:
                                tup = (hue1, hue2)
                                tup_rev = (hue2, hue1)
                                if tup not in seen and tup_rev not in seen:
                                    combinations.append(tup)
                                    seen[tup] = True
                    # tup = (color1, color2)
                    # if tup not in seen:
                    #     combinations.append(tup)
                    #     seen[tup] = True
    else:
        print(top_colors_hues)
        for color1, lst1 in top_colors_hues.items():
            for color2, lst2 in top_colors_hues.items():
                for hue1 in lst1:
                    for hue2 in lst2:
                        if hue1 != hue2:
                            tup = (hue1, hue2)
                            tup_rev = (hue2, hue1)
                            if tup not in seen and tup_rev not in seen:
                                combinations.append(tup)
                                seen[tup] = True
    print("combos")
    print(combinations)
    random.shuffle(combinations)
    combinations = combinations[:30]
    combo_hex_code_lst = []
    for i in range(len(combinations)):
        n = len(combinations)+len(necessary_colors)
        hex_codes = necessary_colors + list(combinations[i])
        print("generate")
        combo_hex_code_lst.append(palette_generator(hex_codes, n, energy))

    # print(combinations)
    return combo_hex_code_lst


def input_to_color(keywords, necessary_colors, energy):
    """ compute the colors from the user input
:param keywords: list of words
:param necessary_color: list of hex codes
    :param energy:
:return: dict with the format {palette_id: [list of hexcodes, ...],...}}
"""
    energy = int(energy)
    # num_colors = int(num_colors)
    palette_dict = {}
    # #print('keywords', keywords)

    # start = time.time()
    top_colors, top_colors_hues = top_colors_from_keywords(keywords, energy)
    #print('TOP COLORS')
    # print(top_colors)
    # #print('top colors time')
    # #print(time.time() - start)

    # start = time.time()
    palettes = create_combinations(
        top_colors, necessary_colors, top_colors_hues, energy)
    # palettes = create_combo_hex_codes(top_colors, necessary_colors, energy)
    # print('PALETTES')
    # print(palettes)
    # #print('combo hex codes time')
    # #print(time.time() - start)

    # palettes is a list of lists - - so loop and change for necesarrycolors
    # #print("LOOK HERE ----------")
    # #print(palettes)
    for x in range(len(palettes)):
        for y in range(len(necessary_colors)):
            palettes[x][y] = necessary_colors[y]
    # #print()
    # print(palettes)
    index = 0
    for p in palettes:
        palette_dict[index] = p
        index += 1

    return palette_dict, top_colors


def getPalettes(keywords, reqColors, energy):
    """
    Returns a list of palettes sorted from highest to lowest ranked.
    Returns a dictionary of the following format: (all clean hexcodes)
        { paletteID: (List of Colors in Palette, Score),
          ...
          paletteID: (List of Colors in Palette, Score) }
    Returns another dictionary that also includes the keyword and the corresponding percentage score
      for each palette

    Params: keywords    List of user's keywords where each string is formatted:
                            'word - definition' [List of Strings]
            reqColors   list of user-inputted clean hexcode color [List of String]
            energy      user-input on the muted to bright scale [Int]
            numColors   number of colors the user wants in their palette [Int]
    """
    cymKeywords, wordMatch = keywordMatch(keywords)

    palettes, top_colors = input_to_color(cymKeywords, reqColors, energy)

    keywords = [i[0] for i in cymKeywords]

    scored, keywordBreakdown = scorePalettes(palettes, keywords, reqColors,
                                             top_colors, wordMatch)

    ranked = sorted(
        scored.items(), key=lambda scored: scored[1][1], reverse=True)

    print("\nGENERATED")
    print(ranked)

    sortedScored = []
    i = 0
    if int(energy) < 2:
        thresh = 130
    elif int(energy) < 9:
        thresh = 115
    else:
        thresh = 105

    while len(sortedScored) < 5 and i < len(ranked):
        tup = ranked[i]
        tooClose = False

        for pal in sortedScored:
            print(tup[0], ',', pal[0])
            if isClose(tup[1][0], pal[1][0], thresh):
                tooClose = True

        if not tooClose:
            shuffled = tup[1][0]
            random.shuffle(shuffled)
            new_tup = (tup[0], (shuffled, tup[1][1]))
            sortedScored.append(new_tup)

        i += 1

    print("\nKEPT")
    print(sortedScored)

    return sortedScored, keywordBreakdown


def scorePalettes(palettes, keywords, reqColors, top_colors, wordMatch):
    """
    Returns a new dictionary that scores and ranks each palette based on the
        following factors and weights:
        1. RGB Euclidian Distance to Required Color                        25%
        2. HSV Cartesian Distance to Required Color                        25%
        3. Delta-E Distance to Required Color?                             0%
        4. Cymbolism Keyword Close Color Percentages                       50%
    Returns another dictionary that also includes the keyword and the corresponding percentage score
      for each palette
    Final dictionary will be of the following format: (all clean hexcodes)
        { paletteID: (List of Colors in Palette, Score),
          ...
          paletteID: (List of Colors in Palette, Score) }
    Params: palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
            keywords    Cymbolism words matched to user input [List of Strings]
            reqColors   list of user-inputted clean hexcode color [List of String]
    """
    # preallocate
    scoreDict = {}
    rgbDists = {}
    hsvDists = {}
    percDists = {}

    # weights
    if len(reqColors) == 0:
        rgbW = .15
        hsvW = .15
        percW = .15
        keyW = .3
        compW = .25
    else:
        rgbW = .1
        hsvW = .1
        percW = .1
        keyW = .3
        compW = .4

    # maximums
    maxRGB = 0
    maxHSV = 0
    maxPerc = 0

    # averages across top colors
    for rc in top_colors:
        # rc = rc[0]
        rgb = getRGBDists(rc, palettes)
        hsv = getHSVDists(rc, palettes)
        perc = getPerceptualDists(rc, palettes)

        for id, dist in rgb.items():
            if dist > maxRGB:
                maxRGB = dist
            if id not in rgbDists:
                rgbDists[id] = dist/len(top_colors)
            else:
                rgbDists[id] += dist/len(top_colors)

        for id, dist in hsv.items():
            if dist > maxHSV:
                maxHSV = dist
            if id not in hsvDists:
                hsvDists[id] = dist/len(top_colors)
            else:
                hsvDists[id] += dist/len(top_colors)

        for id, avg in perc.items():
            if avg > maxPerc:
                maxPerc = avg
            if id not in percDists:
                percDists[id] = avg/len(top_colors)
            else:
                percDists[id] += avg/len(top_colors)

    keywordAvgs, keywordBreakdown = keyword(keywords, palettes, wordMatch)

    complement = {}
    for id, pal in palettes.items():
        minDiff = 1000
        for c1 in pal:
            c1 = convertColor(c1, 'hex', 'rgb')
            for c2 in pal:
                c2 = convertColor(c2, 'hex', 'rgb')
                if c1 != c2:
                    diff = colorDiff(c1, c2, 'rgb')
                    if diff < minDiff:
                        minDiff = diff
        complement[id] = minDiff

    maxDiff = max(complement.values())

    # weighted average of scores
    for id, palette in palettes.items():
        score = 0
        if (rgbDists != {} and maxRGB != 0):
            score += (1 - rgbDists[id]/maxRGB)*100*rgbW
        if (hsvDists != {} and maxHSV != 0):
            score += (1 - hsvDists[id]/maxHSV)*100*hsvW
        if (percDists != {} and maxPerc != 0):
            score += (1 - percDists[id]/maxPerc)*100*percW
        if (keywordAvgs != {}):
            score += keywordAvgs[id]*100*keyW
        if (complement != {}):
            score += (complement[id]/maxDiff)*100*compW

        # CSVpalette = (' '.join([str(elem)
        #                         for elem in palette])).replace(",", " ")

        print("score" , score)
        with open('data/votes.csv', 'r', newline='') as file:
            myreader = csv.reader(file, delimiter=',')
            for rows in myreader:
                if not rows[0] == "Palette":
                    palette1 = rows[0].split(" ")
                    if isClose(palette1, palette, 200):
    
                        for query,tup in wordMatch.items():
                            votes = rows[cymWordsInvInd[tup[0]]]
                            total = int(votes[:votes.find(' ')])
                            if not total== 0:
                                net = int(votes[votes.find(' ')+1:])
                                score += (100 - score)*(net/total)*tup[1]

        # score *= net_updown/total_updown
        print(score)

        scoreDict[id] = (palette, score)

    return scoreDict, keywordBreakdown

def getReqColors(color1, color2):
    reqColors = []
    if color1 and color1 != "":
        reqColors.append(color1.upper())
    if color2 and color2 != "":
        reqColors.append(color2.upper())
    return reqColors

def setupForCsv(voteAndPaletteLst, keywords):
    votes = []
    palettes = []
    for i, voteAndPalette in enumerate(voteAndPaletteLst):
        m = len(voteAndPalette)
        if(not voteAndPalette == ""):
            if(voteAndPalette[m-2]=='-'):
                votes.append(voteAndPalette[-2:])
                voteAndPalette = voteAndPalette[:-3]
        
                palettes.append(voteAndPalette.split(","))
            else:
                votes.append(voteAndPalette[m-1])
                voteAndPalette = voteAndPalette[:-2]
                palettes.append(voteAndPalette.split(","))

    for i in range(len(palettes)):
        print(palettes[i])
        print(votes[i])
        if(not palettes[i]=="" and (int(votes[i]) == 1 or int(votes[i]) == -1)):
            paletteToCSV(palettes[i], keywords, votes[i])

@irsystem.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        errors = []
        keywords = request.form.get("keywords")
        keywordDefs = []
        keywordDefDict = {}
        energy = request.form.get("energy")
        color1 = request.form.get("color1")
        color2 = request.form.get("color2")
        

        submit = True

        if request.form.get('submit-button') == 'vote':
            sortedScored = request.form.get("sortedScored")
            voteAndPaletteLst = request.form.get("votes")
            keywordBreakdown = request.form.get("keywordBreakdown")
            #keywordBreakdown={0: [('beach', 'beach', 59.26734239322961), ('cool', 'cold', 75.88183700692531)], 1: [('cool', 'cold', 60.63718348112445), ('beach', 'beach', 77.6741907104524)], 2: [('cool', 'cold', 57.686154394631124), ('beach', 'beach', 74.90704815062111)], 3: [('cool', 'cold', 65.86606260520283), ('beach', 'beach', 81.09119250781961)], 4: [('cool', 'cold', 54.66467633790308), ('beach', 'beach', 82.34898458047067)], 5: [('cool', 'cold', 54.66467633790297), ('beach', 'beach', 79.59232362124457)], 6: [('cool', 'cold', 65.46685177387367), ('beach', 'beach', 70.67248150603079)], 7: [('cool', 'cold', 65.51381775403013), ('beach', 'beach', 67.66426213227481)], 8: [('cool', 'cold', 70.99318210560921), ('beach', 'beach', 77.87334112195532)], 9: [('cool', 'cold', 61.5843307476117), ('beach', 'beach', 70.90307671935022)], 10: [('cool', 'cold', 54.23415485313615), ('beach', 'beach', 79.83340043516927)], 11: [('cool', 'cold', 64.64494712113688), ('beach', 'beach', 69.87587986001874)], 12: [('cool', 'cold', 69.40416644365125), ('beach', 'beach', 87.66315608741903)], 13: [('cool', 'cold', 56.590281524315216), ('beach', 'beach', 80.91300529752775)], 14: [('cool', 'cold', 55.650961921187374), ('beach', 'beach', 78.18778914011818)], 15: [('cool', 'cold', 71.15756303615665), ('beach', 'beach', 74.4248945227715)], 16: [('cool', 'cold', 55.50223631735883), ('beach', 'beach', 71.2489695393291)], 17: [('cool', 'cold', 60.43366423378012), ('beach', 'beach', 75.0013825560699)], 18: [('cool', 'cold', 63.063759122538016), ('beach', 'beach', 85.0637191372745)], 19: [('cool', 'cold', 63.88566377527488), ('beach', 'beach', 83.1560678270878)], 20: [('cool', 'cold', 50.265529529920784), ('beach', 'beach', 76.19628502508809)]},
            keywordsVote = set()
            print("KEY",keywordBreakdown)
        
            for word  in cymbolism:
                if word in keywordBreakdown :
                    keywordsVote.add(word)
                    
            keywordsVote = list(keywordsVote)
            print(keywordsVote)
            if not voteAndPaletteLst == None :
                print("CSV", voteAndPaletteLst.split(":"), keywordsVote)
                if(not voteAndPaletteLst.split(":")==[""]):
                    setupForCsv(voteAndPaletteLst.split(":"), keywordsVote)
           

        # print('HERE')
        # print(color1)
        # print(color2)
        # print(energy)
        # print(invalidWords)
        # print(keywords)
        # print(multiDefs)

        if request.form.get('submit-button') == 'regenerate' or request.form.get('submit-button') == 'vote':
            print("regenerate")
            keywordDefDict = request.form.get("keywordDefs")
            keywordDefDict = dict((x.strip().replace("'", ""), y.strip().replace("'", ""))
                        for x, y in (element.split(':')
                                for element in keywordDefDict[1:len(keywordDefs)-1].split(', ')))
            keywords = ""
            keywordDefs = []
            for word, d in keywordDefDict.items():
                if keywords != "": keywords += ","
                keywords += word 
                if d == "":
                    keywordDefs.append(word)
                else:
                    keywordDefs.append(word + " - " + d)

            reqColors = getReqColors(color1, color2)
            sortedScored, keywordBreakdown = getPalettes(
                keywordDefs, reqColors, energy)
            return render_template('search.html', netid=netid, sortedScored=sortedScored, keywordBreakdown=keywordBreakdown, keywordDefs=keywordDefDict, keywords=keywords, energy=energy, color1=color1, color2=color2, submit=submit)

        multiDefs = request.form.get("multiDefs")
        invalidWords = request.form.get("invalidWords")
        results = None
        showModal = False

        if (keywords is None and energy is None and color1 is None and color2 is None):
            submit = False

        if keywords is None or len(keywords) == 0 :
            errors.append("keywords1")
        if keywords:
            keywordsList = keywords.split(",")
            if len(keywordsList) == 0:
                errors.append("keywords1")
            elif len(keywordsList) > 5:
                errors.append("keywords2")
            elif invalidWords:
                invalidWordsList = invalidWords.split(",")
                for w in invalidWordsList:
                    if w not in keywordsList:
                        invalidWordsList.remove(w)

                if len(invalidWordsList) > 0:
                    # print("hello?????")
                    errors.append("keywords3")

        if color1 and re.search("^([A-Fa-f0-9]{6})$", color1) is None:
            errors.append("color1")
        if color2 and re.search("^([A-Fa-f0-9]{6})$", color2) is None:
            errors.append("color2")

        multiDefList = [] 

        if request.form.get('submit-button') == 'definitions':
            multiDefList = multiDefs.split(",")
            for d in multiDefList:
                print(d)
                try:
                    if request.form[d]:
                        keywordDefs.append(
                            d + " - " + request.form[d].replace("%", " "))
                        keywordDefDict[d] = request.form[d].replace("%", " ")
                except:
                    print("failed to get form element")
                    print(submit)
                    if len(errors) == 0:
                        errors.append("multi")
                        print("return 1")
                        return render_template('search.html', netid=netid, results=None, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit, multiDefs=multiDefs, showModal=True)

        if request.form.get('submit-button') == 'general' and multiDefs:
            print("return 2")
            return render_template('search.html', netid=netid, results=None, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit, multiDefs=multiDefs, showModal=True)
        
        for k in keywords.split(","):
            if k not in multiDefList:
                keywordDefs.append(k)
                keywordDefDict[k] = ''

        reqColors = getReqColors(color1, color2)
        # reqColors = []
        # if color1:
        #     reqColors.append(color1.upper())
        # if color2:
        #     reqColors.append(color2.upper())

        # display results if no errors
        sortedScored = []
        keywordBreakdown = {}
        if len(errors) == 0:
            sortedScored, keywordBreakdown = getPalettes(
                keywordDefs, reqColors, energy)
            

            print("return 3")
            return render_template('search.html', netid=netid, sortedScored=sortedScored, keywordBreakdown=keywordBreakdown, keywordDefs=keywordDefDict, keywords=keywords, energy=energy, color1=color1, color2=color2, submit=submit)

        # display errors + sticky values
        print("return 4")
        return render_template('search.html', netid=netid, sortedScored=None, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit)

    if request.method == "GET":
        print("return 5")
        return render_template('search.html', netid=netid)
    # return render_template('search.html', netid=netid, sortedScored=[(5, (['0FF4F3', 'F6DA0D', 'DDC114', 'C44D2A', 'BD2456'], 9.116932314670592)), (3, (['F9B00C', '0CF1F0', '79E6A2', 'A17B1E', 'D51F36'], 9.10481916108276)), (1, (['FDF606', '08FAF8', 'BBD4E5', '6E3F55', '16121F'], 1.6377623563612191)), (4, (['FBA10C', 'FDDA0B', 'AEA417', '153F2F', '131D29'], 1.5999848744714602)), (0, (['FCFA0A', 'FBA50C', 'E8321C', '1A2124', '15222E'], 1.5962150584780885))], keywordBreakdown={0: [('beach', 'beach', 2.6857654431513), ("cool", "cool", 7.938)], 1: [('beach', '', 2.6857654431513)], 2: [('beach', '', 2.6857654431513)], 3: [('beach', '', 21.366756192181)], 4: [('beach', '', 2.6857654431513)], 5: [('beach', '', 21.366756192181)]}, keywords="beach", keywordDefs={"beach": None, "cool": "to lose heat"})

    # return render_template('search.html', netid=netid,
    #     sortedScored=[(5, (['0FF4F3', 'F6DA0D', 'DDC114', 'C44D2A', 'BD2456'], 9.116932314670592)), (3, (['F9B00C', '0CF1F0', '79E6A2', 'A17B1E', 'D51F36'], 9.10481916108276)), (1, (['FDF606', '08FAF8', 'BBD4E5', '6E3F55', '16121F'], 1.6377623563612191)), (4, (['FBA10C', 'FDDA0B', 'AEA417', '153F2F', '131D29'], 1.5999848744714602)), (0, (['FCFA0A', 'FBA50C', 'E8321C', '1A2124', '15222E'], 1.5962150584780885))],
    #     keywords="beach", keywordDefs={"beach": "", "cool": "to lose heat"},
    #     color1="000000",
    #     energy=5)
