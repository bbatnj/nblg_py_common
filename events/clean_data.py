import pandas as pd
import numpy as np

class CleanData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_parquet(file_path)
        self.df.index = self.df.index.set_levels(
            pd.to_datetime(self.df.index.levels[1]).tz_convert(None), level=1
        ) 
        self.obtain_all_df_dict()
        self.obtain_multi_event_df()
        
        
    def obtain_all_df_dict(self):
        df_dict = {key: group.droplevel(0) for key, group in self.df.groupby(level=0)}
                # 处理 df_dict: 替换 -inf 和 inf 为 NaN，并删除全 NaN 的 DataFrame
        df_dict = {
            key: df.replace([np.inf, -np.inf], np.nan)  # 替换 inf 和 -inf 为 NaN
            for key, df in df_dict.items()
        }

        # 过滤掉全 NaN 的 DataFrame
        df_dict = {key: df.dropna(how='all') for key, df in df_dict.items() if not df.isna().all().all()}
        self.df_dict = df_dict
        self.all_event_list = list(df_dict.keys())
    
    def obtain_multi_event_df(self):
        ###TODO:这里应当用DSU来处理，不过先用笨方法
        new_dict = {}

        for key, df in self.df_dict.items():
            current_index_set = set(df.index)  # 当前 df 的索引集合
            added = False  # 标记是否已经加入某个 group

            for existing_indices in list(new_dict.keys()):  # 遍历已有的 group
                if current_index_set & set(existing_indices):  # 检查是否有交集
                    new_dict[existing_indices][key] = df  # 添加到已有 group
                    new_dict[existing_indices]['_index_set'].update(current_index_set)  # 更新索引集合
                    added = True
                    break  # 只加入一个 group 即可

            if not added:  # 如果没有匹配的 group，就创建新的
                new_dict[tuple(current_index_set)] = {key: df, '_index_set': current_index_set}

        # for key in list(new_dict.keys()):
        #     new_dict[key].pop('_index_set') # 删除索引集合
        self.multi_key_list = []
        self.multi_df_dict = new_dict
        for key in new_dict.keys():
            #里面有两个event
            if len(new_dict[key])>2:
                self.multi_key_list.append(key)
    
    def obtain_multi_event_point_df_pairs(self, i):
        multi_df = self.multi_df_dict[self.multi_key_list[i]]
        index = list(multi_df['_index_set'])  # 事件索引集合
        point_df_pairs = {}
        not_conclude_df = {}

        for j in range(len(index)):
            # 只提取 index[j] 处的数据
            point_df_pairs[j] = {
                key: multi_df[key].loc[[index[j]]]
                for key in multi_df.keys() if key != '_index_set' and index[j] in multi_df[key].index
            }

            # 删除 index[j] 之后的 DataFrame
            not_conclude_df[j] = {
                key: multi_df[key].drop(index[j], errors='ignore')  # 删除 index[j]，如果不存在不会报错
                for key in multi_df.keys() if key != '_index_set'
            }

        return point_df_pairs, not_conclude_df
