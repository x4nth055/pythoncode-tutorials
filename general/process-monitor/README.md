# [How to Make a Process Monitor in Python](https://www.thepythoncode.com/article/make-process-monitor-python)
to run this:
- `pip3 install -r requirements.txt`
- 
    ```
    python3 process_monitor.py --help
    ```
    **Output:**
    ```
    usage: process_monitor.py [-h] [-c COLUMNS] [-s SORT_BY] [--descending] [-n N]

    Process Viewer & Monitor

    optional arguments:
    -h, --help            show this help message and exit
    -c COLUMNS, --columns COLUMNS
                            Columns to show, available are name,create_time,cores,
                            cpu_usage,status,nice,memory_usage,read_bytes,write_by
                            tes,n_threads,username. Default is name,cpu_usage,memo
                            ry_usage,read_bytes,write_bytes,status,create_time,nic
                            e,n_threads,cores.
    -s SORT_BY, --sort-by SORT_BY
                            Column to sort by, default is memory_usage .
    --descending          Whether to sort in descending order.
    -n N                  Number of processes to show, will show all if 0 is
                            specified, default is 25 .

    ```
## Examples:
- Showing 10 processes sorted by create_time in ascending order:
```
python3 process_monitor.py --sort-by create_time -n 10
```
**Output:**
```
             name  cpu_usage memory_usage read_bytes write_bytes    status          create_time  nice  n_threads  cores
pid
1         systemd        0.0     187.92MB   242.47MB     27.64MB  sleeping  2019-09-09 10:56:21     0          1      4
19   kworker/1:0H        0.0        0.00B      0.00B       0.00B      idle  2019-09-09 10:56:21   -20          1      1
17    ksoftirqd/1        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
16    migration/1        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
15     watchdog/1        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
13        cpuhp/0        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
12     watchdog/0        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
11    migration/0        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
14        cpuhp/1        0.0        0.00B      0.00B       0.00B  sleeping  2019-09-09 10:56:21     0          1      1
9       rcu_sched        0.0        0.00B      0.00B       0.00B      idle  2019-09-09 10:56:21     0          1      4
```
- Showing 20 processes with only name, cpu_usage, memory_usage and status as columns, sorted by memory_usage in descending order:
```
python3 process_monitor.py --columns name,cpu_usage,memory_usage,status -n 20 --sort-by memory_usage --descending
```
**Output:**
```
                name  cpu_usage memory_usage    status
pid
1312          mysqld        0.0     144.63MB  sleeping
915      gnome-shell        0.0      81.00MB  sleeping
3214         python3        0.0      58.12MB   running
1660   rtorrent main        0.0      35.84MB  sleeping
2466   rtorrent main        0.0      24.02MB  sleeping
3186             php        0.0      19.58MB  sleeping
737             Xorg        0.0      15.52MB  sleeping
1452         apache2        0.0      12.18MB  sleeping
872      teamviewerd        0.0      11.53MB  sleeping
974        gsd-color        0.0       8.65MB  sleeping
553   NetworkManager        0.0       7.71MB  sleeping
1045          colord        0.0       7.16MB  sleeping
982     gsd-keyboard        0.0       6.23MB  sleeping
969    gsd-clipboard        0.0       6.09MB  sleeping
548     ModemManager        0.0       5.68MB  sleeping
986   gsd-media-keys        0.0       4.94MB  sleeping
1001       gsd-power        0.0       4.72MB  sleeping
962    gsd-xsettings        0.0       4.59MB  sleeping
1023       gsd-wacom        0.0       4.40MB  sleeping
961      packagekitd        0.0       4.31MB  sleeping
```