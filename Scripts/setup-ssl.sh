#!/usr/bin/env bash

# brew-installed OpenSSL on MacOS
PATH="/opt/homebrew/opt/openssl@3/bin:$PATH"

n=${1:-4}
ssl_dir=${2:-"Player-Data"}

test -e $ssl_dir || mkdir $ssl_dir

echo Setting up SSL for $n parties

for i in `seq 0 $[n-1]`; do
    openssl req -newkey rsa -nodes -x509 -out $ssl_dir/P$i.pem -keyout $ssl_dir/P$i.key -subj "/CN=P$i"
done

# 使用 -f 忽略文件不存在的错误
rm -f $ssl_dir/*.0
# 使用 openssl rehash 替代坏掉的 c_rehash 脚本
openssl rehash -v $ssl_dir