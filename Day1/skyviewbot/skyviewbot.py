#!/usr/bin/env python
# ASTERICS-OBELICS Good Coding Practices (skyviewbot.py)
# V.A. Moss (vmoss.astro@gmail.com), with suggestions from T.J. Dijkema

__author__ = "Stefano Mandelli"
__date__ = "$08-apr-2019 12:00:00$"
__version__ = "0.1"

import os
import sys
from modules.functions import *
from argparse import ArgumentParser, RawTextHelpFormatter
import aplpy
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set matplotlib plotting parameters
# This is because the defaults are sub-optimal
# Maybe you can think of a better way to handle these parameters ;)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

###################################################################

def skyviewbot():
	#
	ar = 0.0
	dec = 0.0
	# Set up argument parsing

	# Make sure you use meaningful variable names!
	# Arguments should include:
	# - option to download from Skyview or use existing FITS file (included)
	# - custom field name
	# - survey to download from Skyview
	# - position to centre the field on    # done
	# - field of view in degrees
	# - coordinate system for the image
	# - output image name to save as a JPEG
	# - colormap to use for imaging (optional!)
	# - option to include a colorbar or not (optional!)
	# - Slack ID of the user making the post
	# - custom colour to post to Slack (optional!)
	# - text to accompany the image posted to Slack (optional!)

	parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
	### ARGUMENTS HERE!! ###
	parser.add_argument('-s', '--skyview',
						action = 'store_true',
						help='Specify whether to download a region from Skyview (default: %(default)s)')
	parser.add_argument('-f', '--fits_name',
						default=None,
						type=str,
						help='Specify name of a custom fits file to use as input (default: %(default)s)')
	parser.add_argument('-fi', '--field', default='PKS1657-298', type=str, help="Specify the field (default: %(default)s) " )
	parser.add_argument('-sur', '--survey', default='DSS', type=str, help='Specify the survey (default: %(default)s)')
	parser.add_argument('-ar', '--rightascension', default=255.291, type=float, help="Specify the Right Ascension of the object (default: %(default)s) ")
	parser.add_argument('-dec', '--declination', default=-29.911, type=float, help="Specify the declination of the object (default: %(default)s)")
	parser.add_argument('-fov', '--fieldOfView', default=1, type=float, help="Specify the Field of View (default: %(default)s)")
	parser.add_argument('-e', '--epoch', default="J2000", type=str, help="Specify the coordinate epoch (default: %(default)s)")

	args = parser.parse_args()

	###################################################################

	# Download an image of choice or use existing one

	# This section should be able to handle querying Skyview, using a custom FITS or the included one
	# Some possible ways to improve:
	# - maybe the region cutout has different width and height dimensions?
	# - maybe the user wants to select on wavelength type, e.g. radio, optical, etc
	# - what if the FITS file of choice is online somewhere, to be downloaded?
	# - what if the user can't get Java working, can you provide an astroquery.Skyview alternative?

	if args.skyview:
		# All parameters in this should be set properly using argparse
		# e.g. call_skyview(args.field, args.survey, args.pos, args.fov, args.coord)
		fits_name = call_skyview(args.field, args.survey, (args.rightascension, args.declination), args.fieldOfView, args.epoch)
	elif args.fits_name:
		fits_name = args.fits_name
	else:
		fits_name = 'results/Skyview_'+args.field+'_'+args.survey+'.fits'

	# This shouldn't be hardcoded, but set as an input argument in the parser above
	field = args.field

	###################################################################

	# Make an image using aplpy

	# Modify the below to be a function call in modules/functions.py
	# Make sure you include docstrings with your function to explain what it does!
	# Use the examples in functions.py to guide your docstring
	# With the following as options (add more if you want to):
	# - fits_name (file name of FITS image)
	# - colormap name
	# - optional to include a colorbar or not
	# - img_name (name of the produced output file)

	# Current parameters
	cmap_name = 'Spectral'
	colorbar = True # As an Aussie, I really struggle with American spelling in Python
	img_name = args.field+".jpg"

	# Construct the figure
	f = aplpy.FITSFigure("results/"+fits_name,figsize=(10,8))
	plt.title(field)
	f.show_colorscale(cmap=cmap_name,stretch='linear')
	f.ticks.set_color('k')
	if colorbar:
		f.add_colorbar()

	# Note: bbox_inches='tight' gets rid of annoying white space, very useful!
	plt.savefig('results/'+img_name,dpi=200,bbox_inches='tight')

	###################################################################

	# Upload the image to Google/Dropbox

	# This is done using a pre-written function included in modules/functions.py
	# Note: you need to login when prompted in the browser
	# With the autoskyview@gmail.com address, not your own!!!
	# See slides for login information
	# Possible way to improve:
	# - what if the user doesn't want to save everything to "results/"?
	# - what happens if something goes wrong with the image upload?

	img_path = 'results/'+img_name
	image_id = upload_to_google(img_path)

	###################################################################

	# Send the results to Slack

	# Modify the below to have these parameters set by your argparse arguments
	# Specifically (add more options if you want to):
	# - msg_color: colour of the message side
	# - msg_text: text to accompany the post
	# - field: name of the field in your image
	# - slack_id: your Slack ID
	# Note: if you add more options, you need to modify also send_to_slack()

	msg_color ='#3D99DD' # Little known fact: this colour is known as Celestial Blue
	msg_text = args.field+' is a great deep-sky object!' # 1707.01542
	slack_id = 'UH77P08BY' # This should be your own Slack ID, if you're testing the code

	# Check for Slack ID
	send_to_slack(msg_color, msg_text, field, slack_id, image_id)


###################################################################

# This part allows you to run the script as normal, as well as a function form
if __name__ == '__main__':
    skyviewbot()
# python skyviewbot.py -s -fi Galaxy -sur DSS -ar 185.63325 -dec 29.8959 -fov 1 -e J2000
