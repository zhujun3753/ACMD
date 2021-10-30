#* 1.把这个目录变成git可以管理的仓库
# git init

#* 2.把文件添加到版本库中
git add . #* 添加到暂存区里面去，.表示添加当前目录下的全部文件
git commit -m "change"
git push origin main

#* 创建SSH Key
# ssh-keygen  -t rsa -C "zhujun3753@163.com"

# git config --global user.email "zhujun3753@163.com"
# git config --global user.name "zhu jun"