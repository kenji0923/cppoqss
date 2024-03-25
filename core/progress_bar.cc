#include <cppoqss/progress_bar.h>

#include <chrono>
#include <cstdint>

#include <indicators/cursor_control.hpp>

#include <cppoqss/mpi_helper.h>


using namespace indicators;


namespace cppoqss {


const size_t ProgressBar::DefaultPrintLimit = 100;
size_t ProgressBar::PrintLimit = DefaultPrintLimit;
bool ProgressBar::IsPrintLimited = true;

std::unique_ptr<indicators::ProgressBar> ProgressBar::bar;
size_t ProgressBar::Nstep;
size_t ProgressBar::NstepEnd;
size_t ProgressBar::CurrentNormalizedStep;

int ProgressBar::maximum_seconds_without_printing = 3600;

void ProgressBar::InitBar(std::string PrefixText, size_t _Nstep)
{
  if (mpi_helper::is_printing_rank()) {
    size_t PrefixTextLength = 34;
    for (size_t i = 0; i < PrefixTextLength; ++i) {
      PrefixText.append(" ");
    }
    std::string ModPrefixText = PrefixText.substr(0, PrefixTextLength);

    if (!IsPrintLimited) PrintLimit = _Nstep;

    bar.reset(new indicators::ProgressBar{
      option::BarWidth{50},
        option::Start{" ["},
        option::Fill{"█"},
        option::Lead{"█"},
        option::Remainder{"-"},
        option::End{"]"},
        option::PrefixText{ModPrefixText},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::MaxProgress{PrintLimit},
        option::ShowPercentage(true)
    });

    NstepEnd = _Nstep;
    Nstep = 0;
    CurrentNormalizedStep = SIZE_MAX;

    last_printed_time = std::chrono::high_resolution_clock::now();
  }
}

void ProgressBar::RestoreDefaultPrintLimit()
{
  PrintLimit = DefaultPrintLimit;
  IsPrintLimited = true;
}

void ProgressBar::SetPrintLimit(size_t n)
{
  PrintLimit = n;
  IsPrintLimited = true;
}

void ProgressBar::UnsetPrintLimit()
{
  PrintLimit = 0;
  IsPrintLimited = false;
}

void ProgressBar::UpdateAndPrintProgress(size_t _Nstep)
{
  auto timer_now = std::chrono::high_resolution_clock::now();
  size_t NewNormalizedStep = std::round((double)_Nstep/NstepEnd*PrintLimit);

  if (NewNormalizedStep != CurrentNormalizedStep) {
    bar->set_progress(NewNormalizedStep);
    CurrentNormalizedStep = NewNormalizedStep;
  } else if (std::chrono::duration_cast<std::chrono::seconds>(timer_now - last_printed_time).count() > maximum_seconds_without_printing) {
    bar->print_progress();
  }
}

size_t ProgressBar::ProgressStep()
{
  if (mpi_helper::is_printing_rank()) {
    UpdateAndPrintProgress(++Nstep);
    return Nstep;
  } else {
    return 0;
  }
}

void ProgressBar::set_progress(size_t i)
{
  if (mpi_helper::is_printing_rank()) {
    UpdateAndPrintProgress(i);
    Nstep = i;
    last_printed_time = std::chrono::high_resolution_clock::now();
  }
}

void ProgressBar::print_progress()
{
  if (mpi_helper::is_printing_rank()) {
    bar->print_progress();
    last_printed_time = std::chrono::high_resolution_clock::now();
  }
}

std::chrono::high_resolution_clock::time_point ProgressBar::last_printed_time;


} // namespace cppoqss
