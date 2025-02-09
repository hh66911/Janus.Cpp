#pragma once

#include <chrono>

class ModelTimer
{
private:
	ModelTimer() = default;
	ModelTimer(const ModelTimer&) = delete;
	ModelTimer& operator=(const ModelTimer&) = delete;
public:
	static ModelTimer& GetInstance() {
		static ModelTimer global_timer;
		return global_timer;
	}
public:
	enum class TimerType
	{
		Model,
		Layer,
		CopyTensor,
		Compute,
		BuildGraph,
		__Count
	};
private:
	std::array<
		std::chrono::time_point<std::chrono::high_resolution_clock>,
		static_cast<size_t>(TimerType::__Count)
	> start_time_points;
	std::array<std::chrono::milliseconds,
		static_cast<size_t>(TimerType::__Count)> durations;
public:
	void Start(TimerType type) {
		start_time_points[static_cast<size_t>(type)] = std::chrono::high_resolution_clock::now();
	}
	void Stop(TimerType type) {
		auto end_time_point = std::chrono::high_resolution_clock::now();
		durations[static_cast<size_t>(type)] +=
			std::chrono::duration_cast<std::chrono::milliseconds>(
				end_time_point - start_time_points[static_cast<size_t>(type)]);
	}
	std::chrono::milliseconds GetDuration(TimerType type) const {
		return durations[static_cast<size_t>(type)];
	}
	void PrintTimeConsumed(TimerType type) const {
		std::cout << "Time consumed for ";
		switch (type)
		{
		case TimerType::Model:
			std::cout << "model";
			break;
		case TimerType::Layer:
			std::cout << "layer";
			break;
		case TimerType::CopyTensor:
			std::cout << "copying tensor";
			break;
		case TimerType::Compute:
			std::cout << "computing";
			break;
		case TimerType::BuildGraph:
			std::cout << "building graph";
			break;
		default:
			throw std::invalid_argument("Invalid TimerType");
		}
		std::cout << ": " << durations[static_cast<size_t>(type)].count() << "ms\n";
	}
	void PrintTimeConsumedAll() const {
		for (size_t i = 0; i < static_cast<size_t>(TimerType::__Count); ++i)
			PrintTimeConsumed(static_cast<TimerType>(i));
	}
	void ClearAll() {
		for (auto& duration : durations)
			duration = std::chrono::milliseconds(0);
	}
};