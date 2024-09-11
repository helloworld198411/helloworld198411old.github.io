# Group and Authority Management

### 组管理

##### 组说明
- 所有者 User
- 所在组 Group
- 其他组 Other

##### 修改文件所有者
- chown `username` `file name`

##### 修改文件所在组
- chgrp `group name` `file name`


### 权限管理
##### ls -l 查询例 *-rwxrw_r__*
- 第0位表示文件类型
  - `d` 文件夹
  - `-` 普通文件
  - `l` 软连接
  - `c` 字符设备
  - `b` 块设备
- 第1-3位表示User权限
- 第4-6位表示Group权限
- 第7-9位表示Other权限
  - `r` read
    - 对文件: 可读取, 查看
    - 对目录: 可读取, 可ls查看
  - `w` write
    - 对文件: 可修改, 但不可删除(删除文件必须拥有对其所在文件夹的w权限)
    - 对目录: 可修改, 可对目录内创建+删除
  - `e` execute
    - 对文件: 可执行
    - 对目录: 可进入到目录内