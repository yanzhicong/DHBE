import os
import sys
import shutil

import numpy as np
import pandas as pd

import shutil

import matplotlib
matplotlib.use("Agg")

from yattag import Doc
from yattag import indent

import matplotlib.pyplot as plt



##################################################

class Plotter(object):
	'''
		class Plotter
		将训练过程中的数值化输出保存成html以及csv文件进行记录

		用法：
		1. 初始化：
			plotter = Plotter()

		2. 记录：
			plotter.scalar('loss1', step=10, value=0.1)

		3. 输出到html文件：
			plotter.to_html_report('./experiment/plt')
		
		4. 输出所有数据分别到单独的csv文件中：
			plotter.to_csv('./experiment/plt')
	'''

	def __init__(self, args=None):
		self._scalar_data_frame_dict = {}
		self._upper_bound = {}
		self._lower_bound = {}
		self.args = args

	def _check_dir(self, path):
		p = os.path.dirname(path)
		if not os.path.exists(p):
			os.mkdir(p)
		return p

	def set_min_max(self, name, min_val=None, max_val=None):
		if min_val is not None:
			self._lower_bound[name] = min_val
		if max_val is not None:
			self._upper_bound[name] = max_val

	@property
	def scalar_names(self):
		return list(self._scalar_data_frame_dict.keys())

	def get(self, name):
		if name in self._scalar_data_frame_dict:
			return self._scalar_data_frame_dict[name]
		else:
			return None

	def has(self, name):
		return name in self._scalar_data_frame_dict

	def keys(self):
		return list(self._scalar_data_frame_dict.keys())


	def scalar(self, name, step, value, epoch=None):
		if isinstance(value, dict):
			data = value.copy()
			data.update({
				'step' : step
			})
		else:
			data = {
				'step' : step,
				name : value,
			}
		
		if epoch is not None:
			data['epoch'] = epoch
				
		df = pd.DataFrame(data, index=[0])

		if name not in self._scalar_data_frame_dict:
			self._scalar_data_frame_dict[name] = df
		else:
			self._scalar_data_frame_dict[name] = self._scalar_data_frame_dict[name].append(df, ignore_index=True)

	def scalar_probs(self, name, step, value_array, epoch=None):
		data_dict = {}
		for i in range(1, 10):
			thres = float(i) / 10.0
			data_dict[str(thres)] = float((value_array < thres).sum()) / float(len(value_array))
		self.scalar(name, step, data_dict, epoch=epoch)
		self.set_min_max(name, 0.0, 1.0)


	def to_csv(self, output_dir):
		" 将记录保存到多个csv文件里面，csv文件放在output_dir下面。"
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		for name, data_frame in self._scalar_data_frame_dict.items():
			csv_filepath = os.path.join(output_dir, 'scalar_'+name+'.csv')
			data_frame.to_csv(csv_filepath, index=False)

	def from_csv(self, output_dir):
		" 从output_dir下面的csv文件里面读取并恢复记录 "
		csv_name_list = [fn[:-4] for fn in os.listdir(output_dir) if fn.endswith('csv')]
		for name in csv_name_list:
			if name.startswith('scalar_'):
				in_csv = pd.read_csv(os.path.join(output_dir, name+'.csv'))
				self._scalar_data_frame_dict[name[len('scalar_'):]] = in_csv

	def to_zip(self, output_path):
		# base_path, file_name = os.path.split(output_path)
		assert output_path.endswith(".zip")
		if not os.path.exists(output_path[:-4]):
			os.mkdir(output_path[:-4])
		self.to_csv(output_path[:-4])
		shutil.make_archive(output_path, 'zip', output_path[:-4])
		shutil.rmtree(output_path[:-4])


	def write_svg_all(self, output_dir):
		" 将所有记录绘制成svg图片 "
		for ind, (name, data_frame) in enumerate(self._scalar_data_frame_dict.items()):
			min_val = self._lower_bound.get(name, None)
			max_val = self._upper_bound.get(name, None)
			output_svg_filepath = os.path.join(output_dir, name+'.svg')
			plt.figure()
			plt.clf()
			headers = [hd for hd in data_frame.columns if hd not in ['step', 'epoch']]
			if len(headers) == 1:
				plt.plot(data_frame['step'], data_frame[name])
			else:
				for hd in headers:
					plt.plot(data_frame['step'], data_frame[hd])
				plt.legend(headers)
			if min_val is not None:
				plt.ylim(bottom=min_val)
			if max_val is not None:
				plt.ylim(top=max_val)

			plt.grid(axis='y')
			plt.grid(axis='x', which='major')
			plt.grid(axis='x', which='minor', color=[0.9, 0.9, 0.9])
			
			plt.tight_layout()
			plt.savefig(output_svg_filepath)
			plt.close()



	def to_html_report(self, output_filepath):
		" 将所有记录整理成一个html报告 "
		self.write_svg_all(self._check_dir(output_filepath))
		doc, tag, text = Doc().tagtext()
		with open(output_filepath, 'w') as outfile:
			with tag('html'):
				with tag('body'):

					title_idx = 1

					if self.args is not None:
						with tag('h3'):
							text('{}. args'.format(title_idx))
							title_idx += 1
						
						key_list = list(self.args.keys())
						key_list = sorted(key_list)

						with tag('div', style='display:inline-block;width:850px;padding:5px;margin-left:20px'):
							for key in key_list:
								with tag('div', style='display:inline-block;width:400px;'):
									text('{} : {}\n'.format(key, self.args[key]))

					with tag('h3'):
						text('{}. scalars'.format(title_idx))
						title_idx += 1

					data_frame_name_list = [n for n, d in self._scalar_data_frame_dict.items()]
					data_frame_name_list = sorted(data_frame_name_list)

					for ind, name in enumerate(data_frame_name_list):
						with tag('div', style='display:inline-block'):
							with tag('h4', style='margin-left:20px'):
								text('(%d). '%(ind+1)+name)
							doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")

			result = indent(doc.getvalue())
			outfile.write(result)



def img_vertical_concat(images, pad=0, pad_value=255, pad_right=False):
	nb_images = len(images)
	h_list = [i.shape[0] for i in images]
	w_list = [w.shape[1] for w in images]
	if pad_right:
		max_w = np.max(w_list)
		images = [i if i.shape[1] == max_w else np.hstack([i, np.ones([i.shape[0], max_w-i.shape[1]]+list(i.shape[2:]), dtype=i.dtype)*pad_value])
				for i in images]
	else:
		assert np.all(np.equal(w_list, w_list[0]))

	if pad != 0:
		images = [np.vstack([i, np.ones([pad,]+list(i.shape[1:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

	if not isinstance(images, list):
		images = [i for i in images]
	return np.vstack(images)


def img_horizontal_concat(images, pad=0, pad_value=255, pad_bottom=False):
	nb_images = len(images)
	h_list = [i.shape[0] for i in images]
	w_list = [w.shape[1] for w in images]
	if pad_bottom:
		max_h = np.max(h_list)
		images = [i if i.shape[0] == max_h else np.vstack([i, np.ones([max_h-i.shape[0]]+list(i.shape[1:]), dtype=i.dtype)*pad_value])
				for i in images]
	else:
		assert np.all(np.equal(h_list, h_list[0]))

	if pad != 0:
		images = [np.hstack([i, np.ones([i.shape[0], pad,]+list(i.shape[2:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

	if not isinstance(images, list):
		images = [i for i in images]
	return np.hstack(images)
	

def img_grid(images, nb_images_per_row=10, pad=0, pad_value=255):
	ret = []
	while len(images) >= nb_images_per_row:
		ret.append(img_horizontal_concat(images[0:nb_images_per_row], pad_bottom=True, pad=pad, pad_value=pad_value))
		images = images[nb_images_per_row:]
	if len(images) != 0:
		ret.append(img_horizontal_concat(images, pad_bottom=True, pad=pad, pad_value=pad_value))
	return img_vertical_concat(ret, pad=pad, pad_right=True, pad_value=pad_value)



def zoom_img(img, times=5):
	h, w, c = img.shape
	img = img.reshape([h, 1, w, 1, c])
	img = np.tile(img, (1, times, 1, times, 1))
	return img.reshape([h*times, w*times, c])

