#!/usr/bin/env python
# coding=utf-8
import paramiko as pko
"""

ssh=pko.SSHClient()
ssh.set_missing_host_key_policy(pko.AutoAddPolicy())
ssh.connect(hostname="59.78.197.77",port=22,username="zhzhang",password="itcs;789")
_,stdout,stderr=ssh.exec_command("touch myxu.hhhh")
print(stdout.read().decode())
ssh.close()
"""
trans=pko.Transport(("59.78.197.77",22))
trans.connect(username="zhzhang",password="itcs;789")
ssh=pko.SSHClient()
ssh._transport=trans
stdin,stdout,stderr=ssh.exec_command('df -hl')
sftp=pko.SFTPClient.from_transport(trans)
sftp.put(localpath='/home/myxu/soft/network',remotepath='/gpfs/home/zhzhang/myxu/ohoho')
sftp.get(localpath='/home/myxu/ESOI-HDNN-MD_GPU1/ESOI_HDNN_MD/Train/yoyo',remotepath='/gpfs/home/zhzhang/myxu/ohoho')
print (stdout.read().decode())
trans.close()
