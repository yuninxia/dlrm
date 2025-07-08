// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: BSD-3-Clause

// -*-Mode: C++;-*- // technically C99

#ifndef CCT_ADDR_H
#define CCT_ADDR_H

#include "../utilities/ip-normalized.h"

typedef struct cct_addr_t cct_addr_t;

struct cct_addr_t {
  // physical instruction pointer: more accurately, this is an
  // 'operation pointer'.  The operation in the instruction packet is
  // represented by adding 0, 1, or 2 to the instruction pointer for
  // the first, second and third operation, respectively.
  ip_normalized_t ip_norm;
};

//
// comparison operations, mainly for cct sibling splay operations
//

static inline bool
cct_addr_eq(const cct_addr_t* a, const cct_addr_t* b)
{
  return ip_normalized_eq(&(a->ip_norm), &(b->ip_norm));
}

static inline bool
cct_addr_lt(const cct_addr_t* a, const cct_addr_t* b)
{
  if (ip_normalized_lt(&(a->ip_norm), &(b->ip_norm))) return true;
  if (ip_normalized_gt(&(a->ip_norm), &(b->ip_norm))) return false;
  return false;
}

static inline bool
cct_addr_gt(const cct_addr_t* a, const cct_addr_t* b)
{
  return cct_addr_lt(b, a);
}

#define assoc_info_NULL {.bits = 0}

#define NON_LUSH_ADDR_INI(id, ip) {.ip_norm = {.lm_id = id, .lm_ip = ip}}

#endif // CCT_ADDR_H
