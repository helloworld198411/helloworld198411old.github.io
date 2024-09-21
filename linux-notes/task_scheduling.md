# Task Scheduling

### crond任务调度
- crontab *option*
    - -e 编辑crond定时任务
    - -l 查询crond任务
    - -r 删除当前用户的所有crond任务
- service crond restart
  - 重启crond后台
- crond任务的时间规则
    - `minute` `hour` `day of month` `month` `day of week` `job`
    - `*` 所有时间
    - `,` 不连续时间
    - `-` 连续时间
    - `*/n` 每隔n时间 


### at任务调度
- atd 后台守护进程, 一次性定时任务
- at `time`
  - 然后编写任务, 两次Ctrl+D退出
- atq 
  - 查询
- atrm `index` 
  - 删除index编号任务