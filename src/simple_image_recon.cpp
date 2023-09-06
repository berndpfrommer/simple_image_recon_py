// -*-c++-*--------------------------------------------------------------------
// Copyright 2023 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <simple_image_recon_lib/simple_image_reconstructor.hpp>

namespace py = pybind11;

struct EventCD
{
  explicit EventCD(uint16_t xa = 0, uint16_t ya = 0, int8_t pa = 0, int32_t ta = 0)
  : x(xa), y(ya), p(pa), t(ta)
  {
  }
  uint16_t x;
  uint16_t y;
  int8_t p;  // 0 = OFF event,  1 = ON event
  int32_t t;
};

class SimpleImageRecon
{
public:
  using State = simple_image_recon_lib::State;
  SimpleImageRecon(
    uint16_t width, uint16_t height, uint32_t cutoff_period, uint16_t tile_size, double fill_ratio)
  {
    recon_.initialize(width, height, cutoff_period, tile_size, fill_ratio);
  }

  py::array_t<SimpleImageRecon::State> get_state() const
  {
    if (!recon_.getState().empty()) {
      return (
        py::array_t<State>({recon_.getHeight(), recon_.getWidth()}, recon_.getState().data()));
    }
    return (py::array_t<State>());
  }

  size_t get_event_window_size() const { return (recon_.getEventWindowSize()); }
  void update(py::array_t<EventCD> events)
  {
    if (events.ndim() != 1) {
      throw std::runtime_error("events should be 2-d numpy array");
    }
    const EventCD * data = events.data();
    const EventCD * end = data + events.size();
    for (const EventCD * e = data; e < end; e++) {
      recon_.event(e->t, e->x, e->y, e->p);
    }
  }

private:
  simple_image_recon_lib::SimpleImageReconstructor recon_;
};

PYBIND11_MODULE(_simple_image_recon, m)
{
  py::options options;
  options.disable_function_signatures();
  m.doc() = R"pbdoc(
        Plugin for testing simple image recon lib in python
    )pbdoc";

  PYBIND11_NUMPY_DTYPE(SimpleImageRecon::State, L, pbar, numPixActive, numEventsInQueue);
  PYBIND11_NUMPY_DTYPE(EventCD, x, y, p, t);

  py::class_<SimpleImageRecon>(
    m, "SimpleImageRecon",
    R"pbdoc(
        Class for simple image reconstruction.
)pbdoc")
    .def(
      py::init<uint16_t, uint16_t, uint32_t, uint16_t, double>(),
      R"pbdoc(
        SimpleImageRecon(width, height, cutoff_period, activity_tile_size, fill_ratio) -> SimpleImageRecon
)pbdoc")
    .def("update", &SimpleImageRecon::update, R"pbdoc(
        update(events) -> None

        Processes buffer of events
        :param events: array with events in metavision format
        :type events: structured numpy array
)pbdoc")
    .def("get_state", &SimpleImageRecon::get_state, py::return_value_policy::reference, R"pbdoc(
        get_state() -> numpy.ndarray['State']

        Fetches updated state.
        :return: numpy array with state ('L' field is reconstructed brightness)
        :rtype: numpy.ndarray['State'], structured numpy array with fields
)pbdoc")
    .def("get_event_window_size", &SimpleImageRecon::get_event_window_size, R"pbdoc(
        get_event_window_size() -> int

        Fetches number of events in event window
        :return: number of events in event window
        :rtype: int
)pbdoc");
}
