import numpy as np 
import matplotlib.pyplot as plt

import json

import argparse
from argparse import ArgumentParser

def load_data(file_name):
	with open(file_name, 'rt') as f:
		return json.load(f)

def cast_str(s, ts):

    if not isinstance(s, str):
        return s

    if isinstance(ts, (list, tuple)):
        for t in ts:
            c = cast_str(s, t)
            if not isinstance(c, str):
                return c

        return s

    try:
        c = ts(s)
        return c
    except:
        return s


class StoreDict(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict(values))

def plot_m_sigma(data, target_field, batch_size = 1, label='data', sigma=1):

	for i, idx in enumerate(range(0, len(data), batch_size)):

		processed_data = []

		for run in data[idx:idx+batch_size]:

			processed_data.append(
				np.array([ 
					v[target_field]
					for k,v in sorted(
						run['val'].items(),
						key = lambda x : int(x[0]),
					)
				])
			)

		processed_data = np.stack(processed_data)

		m = processed_data.mean(0)
		sigma = sigma * np.std(processed_data, axis=0)

		x = np.arange(m.size)

		plt.plot(x, m, label = f'{label}_{i}')
		plt.fill_between(x, m+sigma, m-sigma, label = f'{label}_{i}_std', alpha = 0.5)

if __name__ == '__main__':

	parser = ArgumentParser()

	parser.add_argument(
		'--file',
		nargs = '+',
		type = lambda x : x if x.endswith('.txt') else x + '.txt',
		required = True,
		help = 'Name of text file with data',
	)

	parser.add_argument(
		'--fun',
		choices = list(filter(lambda x : not x.startswith('__') and not x.endswith('__'), globals().keys())),
		required = True,
		help = 'Name of plot to do'
	)

	parser.add_argument(
		'--show',
		action = 'store_true',
		default = False,
		help = "Set this flag to show plot"
	)

	parser.add_argument(
		'--save',
		type = lambda x : x if x.endswith('.png') else x + '.png',
		default = None,
		help = 'If given this is used as the name in which to store the plot'
	)

	parser.add_argument(
		'--title',
		default = 'plot',
		help = 'Title to give to plot'
	)

	parser.add_argument(
		'--pars',
        type = lambda x : (x.split(':', 1)[0], cast_str(x.split(':', 1)[-1], (int, float))),
        nargs = '+',
        action = StoreDict,
        default = dict(),
        help = 'Extra pars for function (series of key:value)'
    )

	args = parser.parse_args()

	# Load data from all files
	data = [load_data(file_name) for file_name in args.file]

	# Run plot function

	globals()[args.fun](data, **args.pars)

	# Add labels

	plt.title(args.title)

	plt.legend()

	# Show and save

	if args.show:
		plt.tight_layout()
		plt.show()

	if args.save is not None:
		plt.save_fig(args.save, dpi = 400, bbox_inches = 'tight')
