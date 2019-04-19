import os


def get_user_ratings(ratins_file_path):
    """
    get user ratings dict
    :param ratins_file_path:
    :return:
    """

    if not os.path.exists(ratins_file_path):
        return {},{}
    fp = open(ratins_file_path)
    num = 0
    user_rating={}
    user_rating_timelevel={}
    for line in fp:
        if num==0:
            num+=1
            continue
        item = line.strip().split(',')
        if len(item)<4:
            continue
        [userid,itemid,rating,timestamp] = item
        if userid + '_' + itemid not in user_rating_timelevel:
            user_rating_timelevel[userid + '_' + itemid]=[]
            user_rating_timelevel[userid + '_' + itemid].append(rating)
            user_rating_timelevel[userid + '_' + itemid].append(int(timestamp))
        if float(rating)<3.0:
            continue
        if userid not in user_rating:
            user_rating[userid]=[]
        user_rating[userid].append(itemid)
        # if userid+'_'+itemid not in user_rating_level:
        #     user_rating_level[userid+'_'+itemid]=rating
    fp.close()
    return user_rating,user_rating_timelevel


if __name__=="__main__":
    user_rating,user_rating_time=get_user_ratings('data/ratings.txt')
    print(user_rating_time[0],user_rating[0])