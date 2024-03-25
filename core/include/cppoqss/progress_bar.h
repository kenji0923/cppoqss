#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H


#include <chrono>
#include <cstddef>
#include <memory>

#include <indicators/progress_bar.hpp>
#include <indicators/setting.hpp>


namespace cppoqss {


/**
 * TODO: change coding style.
 */
class ProgressBar
{
private:
  static const size_t DefaultPrintLimit;
  static size_t PrintLimit;
  static bool IsPrintLimited;

  static std::unique_ptr<indicators::ProgressBar> bar;
  static size_t Nstep;
  static size_t NstepEnd;
  static size_t CurrentNormalizedStep;

  static void UpdateAndPrintProgress(size_t _Nstep);

public:
  typedef indicators::option::PostfixText PostfixText;

  static int maximum_seconds_without_printing;

  static void InitBar(std::string PrefixText, size_t _Nstep);
  static size_t ProgressStep();
  static indicators::ProgressBar* GetPtr() { return bar.get(); }
  static bool CheckPrintLimited() { return IsPrintLimited; }
  static void RestoreDefaultPrintLimit();
  static void SetPrintLimit(size_t n);
  static void UnsetPrintLimit();

  static void set_progress(size_t i);

  template<class... Args>
  static void set_option(Args&&... args);

  static void print_progress();

private:
  static std::chrono::high_resolution_clock::time_point last_printed_time;
};

template<class... Args>
void ProgressBar::set_option(Args&&... args)
{
  bar->set_option(std::forward<Args>(args)...);
}


} // namespace cppoqss


#endif
