#!/bin/bash

# git init
# git remote add pi git@10.2.125.15:/home/git/huaweiCodeCraft.git

#echo "更新ing"
# git pull
#echo "更新end"

echo "提交ing"

git add .
if [ -n "$1" ]
then
	echo "commit: $1"
	git commit -am "$1"
else
	echo "commit: updata"
	git commit -m "updata"
fi
# git push -f pi master
git push
echo "提交end"
