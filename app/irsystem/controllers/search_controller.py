from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.controllers.IR_main import *

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

def palette_generator(hex, n):
	""" Adds on the “N” values to the list of input colors and outputs the api generated palette 
	need to convert inputs to rgb and outputs to hex
    :param hex: list of hex colors (up to three)
	:param n: number of colors
    :return: list of hex color codes computed from the Colormind API
    """	
	return []



def create_combo_hex_codes(top_keywords_colors, necessary_color):
	""" Calls the palette generator helper
    :param keywords: list of top keywords colors
    :param necessary_color: list of hex codes 
	:param 
    :return: a list of lists of hex color codes
    """	

	return [[]]


def input_to_color(keyword, necessary_color, energy ):
	""" compute the colors from the user input
    :param keywords: list of words 
    :param necessary_color: list of hex codes 
	:param energy: 
    :return: dict with the format {palette_id: [list of hexcodes, ...],...}}
    """
	return 	{}

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



