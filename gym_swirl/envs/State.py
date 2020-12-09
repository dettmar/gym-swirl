from collections import namedtuple

State = namedtuple("State", [
	'positions',
	'orientations',
	'dt',
	'T',
	'O_R',
	'O_P',
	'Delta',
	'DT',
	'DR',
	'Gamma',
	'orientation_sums',
	'leftturns',
	'rightturns'
])