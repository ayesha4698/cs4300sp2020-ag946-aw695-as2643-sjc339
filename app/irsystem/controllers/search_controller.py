from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.controllers.IR_main import *
from quickjs import Function
from colormap import rgb2hex
from PIL import ImageColor
import re
import requests
import json

project_name = "Version 1: Color Palette"
net_id = "Ayesha Gagguturi(ag946)"

def hue_adjuster():
	return 0

def energy_adjust(hex, brightness):
	""" adjusts a hex code so that it’s brighter or more muted
	:param hex: string
	:param brightness: decimal for the percentage of how bright
	Input - 5 * 2 * 10 (test how large increase should be)
	Return: hex code
    """

	return ""

def create_combo_hex_codes(keywords, necessary_color):
	""" Calls the palette generator helper
    :param keywords: list of words 
    :param necessary_color: list of hex codes 
	:param 
    :return: a list of lists of hex color codes
    """	
	return 0

def top_colors_from_keywords(keywords, energy):
	""" Generating each keywords top color from the dataset
    :param keywords: list of words 
	:param energy: string of energy level
    :return: list of top colors with energy adjusted (hex codes)
    """	
	return []


def palette_generator(hex_codes, n):
	""" Adds on the “N” values to the list of input colors and outputs the api generated palette 
	need to convert inputs to rgb and outputs to hex
    :param hex_codes: list of hex colors (up to three)
	:param n: number of colors
    :return: list of hex color codes computed from the Colormind API
    """	
	
	url = "http://colormind.io/api/"

	input_lst = []
	if n <= 3:
		for hex in hex_codes:
			input_lst.append(ImageColor.getcolor(hex, "RGB"))

	for add_color in range(n,5):
		input_lst.append("N")
	

	data = {
		"model" : "default",
		"input" : input_lst
	}
	res = requests.post(url, data=json.dumps(data))

	hex = []
	for rgb in res.json()['result']:
		hex.append(rgb2hex(rgb[0], rgb[1], rgb[2]))

	return hex



def create_combo_hex_codes(top_keywords_color_lst, necessary_color_lst):
	""" Calls the palette generator helper
    :param keywords: list of list top keywords colors
    :param necessary_color: list of list hex codes 
	:param 
    :return: a list of lists of hex color codes
    """	
	combo_hex_code_lst = []
	for i in range(len(top_keywords_color_lst)):
		n = len(top_keywords_color_lst)+len(necessary_color_lst)
		hex_codes = top_keywords_color_lst + necessary_color_lst
		combo_hex_code_lst.append(palette_generator(hex_codes, n))
	return combo_hex_code_lst


def input_to_color(keyword, necessary_color, energy ):
	""" compute the colors from the user input
    :param keywords: list of words 
    :param necessary_color: list of hex codes 
	:param energy: 
    :return: dict with the format {palette_id: [list of hexcodes, ...],...}}
    """
	return 	{}


@irsystem.route("/", methods=["GET", "POST"])
def validate():
	keywords = ""
	energy = ""
	color1 = ""
	color2 = ""
	numcolors = ""

	errors = []
	results = ""

	if request.method == "POST":
	#keywords only singluar words, not phrases
		keywords = request.form["keywords"]
		energy = request.form["energy"]
		color1 = request.form["color1"]
		color2 = request.form["color2"]
		numcolors = request.form["numcolors"]
		
		print(keywords)
		print(energy)
		print(color1)
		print(color2)
		print(numcolors)

		results = "yay"

		if not keywords:
			errors.append("keywords")
		if keywords:
			keywords = keywords.replace(" ", "").split(",")
			if len(keywords) == 0:
				errors.append(keywords)
		if not energy:
			errors.append("energy")

		print(re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color1))
		if color1 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color1) == None:
			errors.append("color1")
		if color2 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color2) == None:
			errors.append("color2")
		if not numcolors:
			errors.append("numcolors")
		
	if len(errors) == 0:
		return render_template('search.html', results=results, keywords=keywords, energy=energy, color1=color1, color2=color2, numcolors=numcolors)
	else:
		keywordString = ", ".join(map(str, keywords))
		print(keywordString)
		return render_template('search.html', errors=errors, keywords=keywordString, energy=energy, color1=color1, color2=color2, numcolors=numcolors)

	

