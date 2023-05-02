#!/bin/python

"""
    viewer.py : Result visualizer script.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Experiment result viewer')

parser.add_argument('--live', default=False, action='store_true',
                    help='Enable live updates')

parser.add_argument('--rate', type=int, default=2,
                    help='Live update frames per second')

parser.add_argument('--window', type=int, default=15,
                    help='Rolling average window size')

parser.add_argument('--no_window', default=False, action='store_true',
                    help='Plot data directly, without rolling averages')

parser.add_argument('--xmax', default=0, type=int, help='Truncate data series to <xmax> entries')
parser.add_argument('--x_label', default=None, help='X axis label')
parser.add_argument('--y_label', default='reward', help='Y axis label')
parser.add_argument('--title', default=None, help='Figure title')
parser.add_argument('--save', default=None, help='Figure file output')
parser.add_argument('--width', default=6,
                    help='Figure width for file export (inches)')
parser.add_argument('--height', default=4,
                    help='Figure height for file export (inches)')
parser.add_argument('--dpi', default=100,
                    help='Figure DPI for file export')
parser.add_argument('--series', default='age',
                    help='X series to plot against')
parser.add_argument('--source', help='Data series to plot', default='results.json')

args = parser.parse_args()

if not args.x_label:
    args.x_label = 'episodes'

with open(args.source, 'r') as f:
    fdata = eval(f.read())

# read test episode series
if 'episodes' in fdata:
    data_X = fdata['episodes']
else:
    data_X = list(range(len(fdata[list(fdata.keys())[0]])))

if args.xmax:
    data_X = data_X[:args.xmax]

data_Y_agent = {}
mean_Y_agent = {}

for k in fdata.keys():
    if k == 'episodes':
        continue

    data_Y = fdata[k]
    mean_Y = None

    if args.window % 2 == 0:
        print('WARNING: rolling average window should be odd')

    if args.xmax:
        data_Y = data_Y[:args.xmax]

    if not args.no_window:
        pfx_Y = [data_Y[0]] * (args.window // 2)
        sfx_Y = [data_Y[-1]] * (args.window // 2)

        cmb_Y = np.concatenate((pfx_Y, data_Y, sfx_Y))
        mean_Y = []

        for i in range(len(data_Y)):
            lb = data_Y[0]
            rb = data_Y[-1]

            start = i - args.window // 2
            total = 0
            count = 0

            for j in range(start, start + args.window):
                if j < 0:
                    total += lb
                elif j >= len(data_Y):
                    total += rb
                else:
                    total += data_Y[j]

                count += 1

            mean_Y.append(total / count)

    data_Y_agent[k] = data_Y
    mean_Y_agent[k] = mean_Y

fig, ax = plt.subplots()

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

ax.set_title(args.title)
ax.set_ylabel(args.y_label)
ax.set_xlabel(args.x_label)

lines = {}
mean_lines = {}

for k in data_Y_agent.keys():
    data_Y, mean_Y = data_Y_agent[k], mean_Y_agent[k]
    data_color = colors.pop(0)

    if args.no_window:
        lines[k], = ax.plot(list(range(len(data_Y))),
                data_Y,
                data_color + '-',
                alpha=0.7,
                label=k)
    else:
        lines[k], = ax.plot(np.array(list(range(len(data_Y)))),
                data_Y,
                data_color + '-',
                alpha=0.25)

        mean_lines[k], = ax.plot(np.array(list(range(len(mean_Y)))),
                mean_Y,
                data_color + '-', alpha=0.7,
                label=k)

ax.legend()

min_y = 0

while True:
    for k in data_Y_agent.keys():
        data_Y, mean_Y = data_Y_agent[k], mean_Y_agent[k]
        
        lines[k].set_data(np.array(list(range(len(data_Y)))), data_Y)

        #min_y = min(min_y, min(data_Y))
        #max_y = max(max_y, max(data_Y))
        #plt.ylim((min(data_Y), max(plt.ylim()[1], max(data_Y))))
        #plt.xlim((min(data_X), max(plt.xlim()[1], max(data_X))))

        if not args.no_window:
            mean_lines[k].set_data(np.array(list(range(len(mean_Y)))), mean_Y)

    if args.live:
        #plt.draw()
        plt.pause(1 / args.rate)

        if not plt.get_fignums():
            break
    else:
        plt.show()
        break

if args.save:
    #ax.set_size_inches(args.width, args.height)
    plt.savefig(args.save, dpi=args.dpi)
