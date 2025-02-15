﻿#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <thread>
#include <locale>

#include "vq_model.h"
#include "language_model.h"
#include "timer.h"
#include "tokenizer.h"

#include "generate.h"

std::string edit_text;
int main(int argc, char** argv)
{
	std::locale::global(std::locale("zh_CN.utf8"));

	// 窗口过程函数
	auto WndProc = [](
		HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam
		) -> LRESULT
		{
			static HFONT hFont = NULL;
			constexpr uint64_t ID_BUTTON = 1;
			constexpr uint64_t ID_EDIT = 2;
			switch (msg) {
			case WM_CREATE: {
				// 创建字体（宋体，四号字，约 14 磅）
				hFont = CreateFont(
					-MulDiv(14, GetDeviceCaps(GetDC(hwnd), LOGPIXELSY), 72), // 字体高度
					0,                      // 字体宽度
					0,                      // 字体倾斜角度
					0,                      // 字体倾斜角度
					FW_NORMAL,              // 字体粗细
					FALSE,                  // 是否斜体
					FALSE,                  // 是否下划线
					FALSE,                  // 是否删除线
					DEFAULT_CHARSET,        // 字符集
					OUT_DEFAULT_PRECIS,     // 输出精度
					CLIP_DEFAULT_PRECIS,    // 裁剪精度
					DEFAULT_QUALITY,        // 字体质量
					DEFAULT_PITCH | FF_SWISS, // 字体间距和字体系列
					L"宋体"                 // 字体名称
				);

				// 创建按钮
				HWND hButton = CreateWindow(L"button", L"Click Me",
					WS_VISIBLE | WS_CHILD,
					125, 270, 200, 50,
					hwnd, (HMENU)ID_BUTTON, NULL, NULL);
				// 设置按钮字体
				SendMessage(hButton, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

				// 创建文本框
				HWND hEdit = CreateWindow(L"edit", L"",
					WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL,
					75, 50, 300, 200,
					hwnd, (HMENU)ID_EDIT, NULL, NULL);
				// 设置文本框字体
				SendMessage(hEdit, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
				break;
			}case WM_COMMAND: {
				// 检查是否是按钮点击事件
				if (LOWORD(wParam) == 1) {
					// 关闭窗口
					PostMessage(hwnd, WM_CLOSE, 0, 0);
				}
				break;
			}
			case WM_DESTROY: {
				// 获取文本框的内容
				HWND hEdit = GetDlgItem(hwnd, 2);
				int len = GetWindowTextLength(hEdit);

				std::wstring text(len + 1, L'\0');
				edit_text.resize(text.length() * 3);
				GetWindowText(hEdit, text.data(), len + 1);
				WideCharToMultiByte(
					CP_UTF8, 0, text.data(), -1,
					edit_text.data(), (int)edit_text.size(),
					NULL, NULL);
				edit_text.resize(strlen(edit_text.c_str()));

				PostQuitMessage(0);
				break;
			}
			default:
				return DefWindowProc(hwnd, msg, wParam, lParam);
			}
			return 0;
		};

	// 注册窗口类
	SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
	WNDCLASS wc = { 0 };
	auto hInstance = GetModuleHandleA(NULL);
	wc.lpfnWndProc = WndProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = L"MyWindowClass";
	// 不使用默认的窗口样式，避免显示边框
	wc.style = 0;

	RegisterClass(&wc);

	// 创建窗口
	HWND hwnd = CreateWindow(wc.lpszClassName, L"Button Window",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 470, 400,
		NULL, NULL, hInstance, NULL);

	if (hwnd == NULL) {
		return 0;
	}

	// 显示窗口
	ShowWindow(hwnd, SW_SHOW);
	UpdateWindow(hwnd);

	std::cout << "请输入文本" << std::endl;

	// 等待窗口关闭
	MSG msg = { 0 };
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	std::cout << "文本: " << edit_text << std::endl;

	SetProcessAffinityMask(GetCurrentProcess(), 0xFFFF);

	constexpr size_t num_imgs = 1;
	constexpr size_t img_sz = 384;
	constexpr size_t num_patchs = img_sz * img_sz / 256;
	auto language_model = std::make_shared<LanguageModel>(true, 30, num_threads);
	auto tokenizer = load_bpe_model(R"(.\Janus-Pro-7B)");
	std::vector<int> input = tokenizer_encode(tokenizer, edit_text);
	const std::vector<int> gen_prefix = { 100000, 5726, 25, 207 };
	input.insert_range(input.begin(), gen_prefix);
	input.push_back(100016);

	auto time_start = std::chrono::high_resolution_clock::now();
	auto embeddings = language_model->preprocess(input, num_imgs, num_patchs);
	std::vector<std::vector<int>> outtokens;
	auto imgs = generate(embeddings, language_model, 1, num_imgs, img_sz, 5, outtokens);
	auto time_end = std::chrono::high_resolution_clock::now();
	auto dur_sec = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);
	std::cout << "用时: " << dur_sec << std::endl;

	for (auto [i, token_ids] : outtokens | std::views::enumerate) {
		std::string decoded = tokenizer_decode(tokenizer, token_ids);
		std::cout << "图片 " << i + 1 << "：" << decoded << std::endl;
	}

	auto output_folder = std::filesystem::current_path() / "generated-imgs";
	for (auto [i, img] : imgs | std::views::enumerate) {
		cv::imwrite((output_folder / ("out" + std::to_string(i) + ".png")).string(), img);
	}

	cv::imshow("output", imgs[0]);
	cv::waitKey();
	return 0;
}
