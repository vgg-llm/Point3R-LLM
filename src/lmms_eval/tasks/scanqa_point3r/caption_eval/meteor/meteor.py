#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR
JAVA_BIN='java'
# meteor-1.5 loads data/paraphrase-en.gz (61MB gzipped) into the heap; -Xmx2G is
# borderline and makes the JVM GC-thrash for minutes-to-forever on memory-tight
# nodes. Override with METEOR_JAVA_XMX if a node needs something else.
JAVA_XMX = os.environ.get('METEOR_JAVA_XMX', '4G')


class Meteor:

    def __init__(self):
        self.meteor_cmd = [JAVA_BIN, '-Xmx{}'.format(JAVA_XMX), '-jar', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        # stderr must NOT be a PIPE: nothing in this wrapper ever drains it, so
        # once the JVM writes ~64KB of warnings the pipe fills, java blocks on
        # stderr, stops producing stdout, and compute_score() blocks forever on
        # readline() -> hard deadlock with no output.
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.DEVNULL)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def _readline(self):
        """Read one line from the JVM, failing loudly if it died.

        An empty read means EOF (the java process exited), which the callers
        would otherwise turn into `float('')` or an endless wait.
        """
        line = self.meteor_p.stdout.readline()
        if not line:
            rc = self.meteor_p.poll()
            raise RuntimeError(
                "METEOR java subprocess produced no output (exit code {}); "
                "cmd was: {}".format(rc, ' '.join(self.meteor_cmd))
            )
        return line.decode().strip()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        try:
            for i in imgIds:
                assert(len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
            self.meteor_p.stdin.flush()
            for i in range(0,len(imgIds)):
                scores.append(float(self._readline()))
            score = float(self._readline())
        finally:
            self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self._readline()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        try:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
            self.meteor_p.stdin.flush()
            stats = self._readline()
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats
            self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
            self.meteor_p.stdin.flush()
            score = float(self._readline())
        finally:
            self.lock.release()
        return score

    def __exit__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.wait()
        self.lock.release()


# Wall-clock ceiling for the whole METEOR pass. The JVM can still GC-thrash on a
# memory-tight node, which looks exactly like a hang; without a ceiling that
# stalls the entire eval *after* inference has finished but *before* any samples
# are written to disk, losing every prediction.
METEOR_TIMEOUT_SEC = float(os.environ.get('METEOR_TIMEOUT_SEC', 900))


def safe_meteor_score(gts, res, timeout=None):
    """Compute METEOR, returning None instead of hanging or raising.

    Returns the usual (score, scores) tuple, or None if java is unavailable,
    the JVM dies, or the pass exceeds `timeout` seconds.
    """
    timeout = METEOR_TIMEOUT_SEC if timeout is None else timeout
    box = {}

    def _run():
        try:
            box['meteor'] = meteor = Meteor()
        except Exception as e:  # java missing, jar unreadable, ...
            box['error'] = e
            return
        try:
            box['result'] = meteor.compute_score(gts, res)
        except Exception as e:
            box['error'] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # Kill the JVM so the blocked readline() raises and the thread unwinds.
        meteor = box.get('meteor')
        if meteor is not None:
            try:
                meteor.meteor_p.kill()
            except Exception:
                pass
        print('METEOR timed out after {}s; skipping it.'.format(timeout), file=sys.stderr)
        return None

    if 'error' in box:
        print('METEOR failed ({}: {}); skipping it.'.format(
            type(box['error']).__name__, box['error']), file=sys.stderr)
        return None

    return box.get('result')
