import math
import time

from tqdm import tqdm

import data_helper


def calc_similar(series1, series2, targetUser, otherUser, user_rating_time_dict, alpha=0.7):
    """
    计算基于的余弦相似度
    :param series1:
    :param series2:
    :return:
    """
    # unionLen = len(set(series1) & set(series2))
    norm_mult = len(series1) * len(series2)
    unionItem = set(series1) & set(series2)
    unionSum = 0
    for u in unionItem:
        u1_time = user_rating_time_dict[str(targetUser) + '_' + u][1] or 0
        u2_time = user_rating_time_dict[otherUser + '_' + u][1] or 0
        unionSum += 1 / (1 + alpha * abs(u1_time - u2_time))
    similarity = unionSum / math.sqrt(norm_mult)
    return similarity


def calc_user_similarity(user_ratings, user_rating_level_time, targetID=1, TopN=10):
    """
    计算targetID的用户与其他用户的相似度
    :param user_ratings:
    :param targetID:
    :param TopN:
    :return:
    """
    user_keys = list(user_ratings.keys())
    # other_user= copy.deepcopy(user_keys)
    if targetID in user_keys:
        user_keys.remove(targetID)
    similarity_dict = {}
    targetitem = user_ratings[str(targetID)] or 0
    for u in user_keys:
        useritem = user_ratings[u]
        similarity = calc_similar(targetitem, useritem, targetID, u, user_rating_level_time)
        if u not in similarity_dict:
            # similarity_dict[targetID+'_'+u]=similarity
            similarity_dict[u] = similarity
    similarity_list = sorted(similarity_dict, key=lambda x: similarity_dict[x], reverse=True)[:TopN]
    similarity_dict_list = list(map(lambda x: {x: similarity_dict[x]}, similarity_list))
    similarity_dict_data = {}
    for u in similarity_dict_list:
        for k, v in u.items():
            similarity_dict_data.setdefault(k, v)

    f = open(f'dict/similarity_dict4({targetID}).txt', 'w+')
    for key in similarity_dict.keys():
        line = key + ',' + str(similarity_dict[key]) + '\n'
        f.writelines(line)
    f.close()
    return similarity_dict


def calc_interest(user_ratings, rating_level_times, similarity_dict, targetItemID=1, alpha=0.7, TopN=10):
    """
    计算目标用户对目标物品的感兴趣程度
    :param user_ratings:
    :param similarity_dict:
    :param rating_level_times:
    :param targetID:
    :param TopN:
    :return:
    """
    similarity_user = similarity_dict.keys()  # 和用户兴趣最相似的K个用户
    similarity_data = {}
    user_inst_item = []  # 用户感兴趣候选集
    for u in similarity_user:  # K个用户数据
        similarity_data.setdefault(u, user_ratings[u])

    similarUserValues = list(similarity_dict.values())  # 用户和其他用户的兴趣相似度
    for u in similarity_user:
        if targetItemID in similarity_data[u]:
            user_inst_item.append(float(rating_level_times[u + "_" + str(targetItemID)][0]))
        else:
            user_inst_item.append(0)

    timesbase = []  # 时间上下文相关
    for u in similarity_user:
        u1_time = rating_level_times.get(u + "_" + str(targetItemID), None)
        if u1_time:
            u1_time = float(u1_time[1])
        else:
            u1_time = int(time.time() - 1000)
        timesbase.append(1 / (1 + alpha * (int(time.time() - u1_time))))
    interest = sum([similarUserValues[v] * user_inst_item[v] * timesbase[v] for v in range(len(similarUserValues))])
    return interest


def calc_recommend_item(user_ratings, user_rating_level_time, targetUserID=1, TopN=10):
    """
    计算推荐给targetUserID的用户的TopN物品
    :param user_ratings:
    :param user_rating_level_time:
    :param targetUser:
    :param TopN:
    :return:
    """
    similarity_dict = calc_user_similarity(user_ratings, user_rating_level_time, targetUserID, TopN)
    target_user_movieId = set(user_ratings[str(targetUserID)])
    other_user_movidI = []
    for u in user_ratings.keys():
        if u != targetUserID:
            other_user_movidI.append(user_ratings[u])
    # other_user_movidI=set(user_ratings.remove(targetUserID))
    other_user_movieIds = []
    for ids in other_user_movidI:
        for id in ids:
            other_user_movieIds.append(id)
    other_user_movidIs = set(other_user_movieIds)
    movieId = list(target_user_movieId ^ other_user_movidIs)  # 差集

    rsList = dict()
    for movie in tqdm(
            movieId,
            position=0,
            leave=True,
            desc="Compute rs-movie",
    ):
        q = calc_interest(user_ratings, user_rating_level_time, similarity_dict, movie)
        rsList[movie] = q

    return sorted(rsList.items(), key=lambda d: d[1], reverse=True)[:TopN]
    # interestList=[calc_interest(user_ratings,user_rating_level_time,similarity_dict,movie) for movie in movieId]
    # return sorted(interestList,key=lambda x:x,reverse=True)[:TopN],similarity_dict

# 基于命令行的推荐预测程序
if __name__ == "__main__":
    # 导入数据
    print('pls wait to load file...')
    user_ratings, user_ratings_level_time = data_helper.get_user_ratings('data/ratings.txt')

    a = list(user_ratings)

    print(f'users info: userid from {min(a,key=lambda x :int(x))} to {max(a,key=lambda x :int(x))}.\n')
    # 预测程序
    var = 1
    while var == 1:
        user = input("pls input want to forcast userid\n(input \"exit\" to exit)...> ")
        if user == 'exit' : break

        if user_ratings.get(user, -1) == -1:
            print('userid not exist!\n')
            continue
        else:
            try:
                topN = int(input('pls input how many items want to forcast...> '))
            except:
                print('pls input number!\n')
                continue
            else:
                print("start calc...")
                start = time.time()
                a = calc_recommend_item(user_ratings, user_ratings_level_time, targetUserID=user, TopN=topN)
                i = 0
                for item in a:
                    i += 1
                    s = str(item)
                    print(f'forcast {i}----{s}')
                print('Cost time: %f' % (time.time() - start))
                continue



    # 离线计算所有用户的预测，用于其它应用
    # print("start calc...")
    # start = time.time()
    # topN = 3
    # for user in user_ratings.keys():
    #     print(f'\n forcast user({user}) Top{topN} likes...')
    #     a = calc_recommend_item(user_ratings, user_ratings_level_time, targetUserID=user, TopN=topN)
    #     print(a)
    #     print('\n')
    #
    # print('Cost time: %f' % (time.time() - start))
