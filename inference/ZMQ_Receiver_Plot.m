% ZMQ_Receiver_Plot.m
% 实时接收 Python 端的 ZMQ 数据并绘图。
% 需要使用 jeromq.jar。如果此前没有加载过，请提前将其加入 javaclasspath
% 或者在命令窗口使用 javaaddpath('/path/to/jeromq.jar') 加入。

function ZMQ_Receiver_Plot()
% ZMQ 配置
port = 5556;
topic = 'INSTRUMENT_INFO';

import org.zeromq.ZMQ;

% 创建 ZMQ 上下文和 Subscriber 套接字
context = ZMQ.context(1);
subscriber = context.socket(ZMQ.SUB);

url = sprintf('tcp://127.0.0.1:%d', port);
fprintf('正在连接至 ZMQ Publisher: %s...\n', url);
subscriber.connect(url);

% 设置订阅的主题
subscriber.subscribe(int8(topic));
% 设置接收超时为 50 毫秒，避免长时间阻塞 UI 刷新
subscriber.setReceiveTimeOut(50);

% 图形界面设置
fig = figure('Name', '实时器械特征追踪', 'Position', [100, 100, 1400, 400]);

% 左侧图：Tip 轨迹
subplot(1, 3, 1);
hPoint = plot(nan, nan, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;
% 仅保留最新的轨迹记录绘制
title('Tip 运动轨迹');
xlabel('X 坐标 (像素)');
ylabel('Y 坐标 (像素)');
grid on;
% 根据摄像头分辨率设定坐标轴（1920x1080）并保持物理长宽比例视觉真实
axis([0 1920 0 1080]);
set(gca, 'YDir', 'reverse'); % 图像坐标系的 Y 轴向下
daspect([1 1 1]); % 强制 X 轴和 Y 轴的数据比例为 1:1，使得绘制出的形状比例符合真实比例

% 中间图：宽度变化趋势
subplot(1, 3, 2);
hWidth = plot(nan, nan, 'r-', 'LineWidth', 1.5);
title('Tip 宽度趋势');
xlabel('采样序列帧序');
ylabel('宽度数值 (像素)');
grid on;

% 右侧图：面积变化趋势
subplot(1, 3, 3);
hArea = plot(nan, nan, 'k-', 'LineWidth', 1.5);
title('掩码物理面积趋势');
xlabel('采样序列帧序');
ylabel('面积 (平方像素)');
grid on;

% 数据缓冲区 (保留最新的 max_pts 个数据点)
max_pts = 300;
tipX_buf = nan(1, max_pts);
tipY_buf = nan(1, max_pts);
width_buf = nan(1, max_pts);
area_buf = nan(1, max_pts);
frame_idx_buf = nan(1, max_pts);

count = 0;

fprintf('开始接收并绘制... (关闭绘图窗口结束程序)\n');

% 只要图形窗口还存在，循环接收
while ishandle(fig)
    try
        % 接收 ZMQ 消息
        recvBytes = subscriber.recv(0);

        if isempty(recvBytes)
            drawnow limitrate;
            continue; % 未收到新消息超时，只刷新 UI
        end

        % 转换为字符串
        msg_str = char(recvBytes');

        if startsWith(msg_str, topic)
            % 提取 JSON 负载部分
            json_str = strtrim(extractAfter(msg_str, topic));

            % 解析 JSON
            data = jsondecode(json_str);

            % 提取数据
            tx = data.tip.x;
            ty = data.tip.y;
            tw = data.width;
            ta = data.area;

            count = count + 1;

            % 滚动更新缓冲区
            if count <= max_pts
                tipX_buf(count) = tx;
                tipY_buf(count) = ty;
                width_buf(count) = tw;
                area_buf(count) = ta;
                frame_idx_buf(count) = count;
                current_len = count;
            else
                tipX_buf = [tipX_buf(2:end), tx];
                tipY_buf = [tipY_buf(2:end), ty];
                width_buf = [width_buf(2:end), tw];
                area_buf = [area_buf(2:end), ta];
                frame_idx_buf = [frame_idx_buf(2:end), count];
                current_len = max_pts;
            end

            % 更新左侧 Tip 轨迹图
            set(hPoint, 'XData', tipX_buf(1:current_len), 'YData', tipY_buf(1:current_len));

            % 更新中间宽度趋势图
            set(hWidth, 'XData', frame_idx_buf(1:current_len), 'YData', width_buf(1:current_len));
            
            % 更新右侧面积趋势图
            set(hArea, 'XData', frame_idx_buf(1:current_len), 'YData', area_buf(1:current_len));

            % 动态调整右侧两图的 X 轴范围，使曲线一直向左滚动
            if count > max_pts
                subplot(1, 3, 2);
                xlim([frame_idx_buf(1), frame_idx_buf(end)]);
                subplot(1, 3, 3);
                xlim([frame_idx_buf(1), frame_idx_buf(end)]);
            end

            drawnow limitrate;
        end

    catch ME
        % 捕获解析或 ZMQ 本身的异常处理
        if strcmp(ME.identifier, 'MATLAB:class:OperationTerminatedByInterrupt')
            break;
        end
    end
end

% 清理资源
fprintf('关闭 ZMQ 连接...\n');
subscriber.close();
context.term();
fprintf('程序已结束。\n');
end
