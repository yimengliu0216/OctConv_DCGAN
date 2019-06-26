#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


fig_time, ax_time = plt.subplots()

num_epoch = 200

dcgan_time = []
dcgan_g_time = []
dcgan_d_time = []
dcgan_gd_time = []

dcgan_time_avg = 0.0
dcgan_g_time_avg = 0.0
dcgan_d_time_avg = 0.0
dcgan_gd_time_avg = 0.0


with open('time/dcgan_time.txt', 'r') as f:
    dcgan_time_str = f.read().split('\n')
    for i in range(num_epoch):
        dcgan_time.append(float(dcgan_time_str[i]))
        dcgan_time_avg += float(dcgan_time_str[i])
    dcgan_time_avg /= num_epoch
        
with open('time/dcgan_time_g.txt', 'r') as f:
    dcgan_g_time_str = f.read().split('\n')
    for i in range(num_epoch):
        dcgan_g_time.append(float(dcgan_g_time_str[i]))
        dcgan_g_time_avg += float(dcgan_g_time_str[i])
    dcgan_g_time_avg /= num_epoch

with open('time/dcgan_time_d.txt', 'r') as f:
    dcgan_d_time_str = f.read().split('\n')
    for i in range(num_epoch):
        dcgan_d_time.append(float(dcgan_d_time_str[i]))
        dcgan_d_time_avg += float(dcgan_d_time_str[i])
    dcgan_d_time_avg /= num_epoch

with open('time/dcgan_time_gd.txt', 'r') as f:
    dcgan_gd_time_str = f.read().split('\n')
    for i in range(num_epoch):
        dcgan_gd_time.append(float(dcgan_gd_time_str[i]))
        dcgan_gd_time_avg += float(dcgan_gd_time_str[i])
    dcgan_gd_time_avg /= num_epoch


ax_time.plot(dcgan_time, label="DCGAN")
ax_time.plot(dcgan_g_time, label="DCGAN_G")
ax_time.plot(dcgan_d_time, label="DCGAN_D")
ax_time.plot(dcgan_gd_time, label="DCGAN_GD")


ax_time.set(xlabel='Epoch', ylabel='Time (s)', title='Training Time')
ax_time.legend()

fig_time.savefig("time/train_time.png")


dcgans_time_avg = [dcgan_time_avg, dcgan_g_time_avg, dcgan_d_time_avg, dcgan_gd_time_avg]
with open('time/dcgans_time_avg.txt', 'a') as f:
        f.write(str(dcgans_time_avg))
