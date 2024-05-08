# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 22:07
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'nyu':
            return 'F:\wp\wp\cs543\\final_project\FCRN_pytorch\data\official_splits'
        elif database == 'kitti':
            return '/home/data/UnsupervisedDepth/wangixn/kitti'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
