import { NextRequest, NextResponse } from 'next/server';
import { del } from '@vercel/blob';
import { z } from 'zod';

const paramsSchema = z.object({
  space_x: z.number(),
  space_y: z.number(),
  space_z: z.number(),
  a_min: z.number(),
  a_max: z.number(),
  b_min: z.number(),
  b_max: z.number(),
  roi_x: z.number(),
  roi_y: z.number(),
  roi_z: z.number(),
  num_samples: z.number(),
});

// Set the max time of the request
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  let url;

  try {
    const formData = await request.formData();
    const file = formData.get('file') as string;
    const selectedTargets = formData.get('selectedTargets') as string;
    const params = JSON.parse(formData.get('params') as string);

    console.log('formData:', formData);
    console.log('file:', file);
    console.log('paramsRaw:', params);
    console.log('selectedTargets:', selectedTargets);

    // JSON 데이터를 파싱
    // const params = JSON.parse(paramsRaw);

    url = file;

    // Validate inputs
    if (!file || !selectedTargets) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Validate parameters using safeParse
    const parsedParams = paramsSchema.safeParse(params);
    if (!parsedParams.success) {
      return NextResponse.json(
        { error: 'Invalid parameters: ' + parsedParams.error.message },
        { status: 400 }
      );
    }

    /**
     * const json = await fetch(...).then(r => r.json());
     * => 서버가 뭐라 응답하는지 관심 없고, JSON으로 그냥 만들어서 줘
     * => 서버가 JSON이 아니게 응답하면 터짐
     *
     * const str = await fetch(...).then(r => r.text());
     * console.log(str);
     * const json = JSON.parse(str);
     * => 서버가 뭐라 응답하든지 아무것도 하지말고 그냥 그 텍스트 그대로 줘
     * => 그담에 그 텍스트를 그대로 로그를 찍고
     * => 그리고 JSON으로 parse 해봐
     */

    const response = await fetch(`${process.env.RUNPOD_ENDPOINT}/runsync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.RUNPOD_ENDPOINT_KEY}`,
      },
      body: JSON.stringify({
        input: {
          ...params,
          url: file,
          targets: JSON.parse(selectedTargets),
        },
      }),
    }).then((r) => r.json());

    const text = await response.text();
    console.log('runpod responded:', text);
    const res = JSON.parse(text);

    if (res.error) {
      throw new Error(
        'An error occurred while processing the image: ' + res.error
      );
    }

    // Return the download URL to the client
    return NextResponse.json(res.output);
  } catch (error) {
    console.error('Error processing CT image:', error);
    return NextResponse.json(
      { error: 'An error occurred while processing the image' },
      { status: 500 }
    );
  } finally {
    if (url) {
      await del(url);
    }
  }
}
